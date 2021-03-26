<?php
header('Content-Type:application/json');

$baseFileName = 'results/' . uniqid();

switch ($_POST['imgMethod']) {
    case 'gallery':
        $fileName = $baseFileName . '.jpg';
        copy($_POST['imgSrc'], $fileName);
        break;
    case 'url':
        $path = parse_url($_POST['imgSrc'], PHP_URL_PATH);
        $extension = pathinfo($path, PATHINFO_EXTENSION);
        $fileName = $baseFileName . '.' . $extension;
        file_put_contents($fileName, file_get_contents($_POST['imgSrc']));
        break;
    case 'local':
        $extension = explode('/', mime_content_type($_POST['imgSrc']))[1];
        $fileName = $baseFileName . '.' . $extension;
        file_put_contents($fileName, base64_decode(
            explode(';base64,', $_POST['imgSrc'])[1]
        ));
        break;
}

$referringExpression = $_POST['referringExpression'];

// TODO: remember to check if it's neccesary to set XDG_CACHE_HOME
$command = '. Code/.venv/bin/activate 2>&1 &&'.
         ' XDG_CACHE_HOME=.cache/ MPLCONFIGDIR=.cache/' .
         ' python Code/comprehend.py' .
         ' --resume Code/checkpoints/model_refcoco.pth' .
         ' --img ' . $fileName .
         ' --sent "' . $referringExpression . '"' .
         ' --device cpu' .
         ' --output ' . $baseFileName . '.out.jpg 2>&1';


exec($command, $outputCommand);

$output = [
    'command' => $command,
    'outputCommand' => $outputCommand,
    'resultImgSrc' => 'api/' . $baseFileName . '.out.jpg'
];

if (isset($_POST['debug'])) {
    print_r($_POST);
    echo $command;
    echo $outputCommand;
    print_r($outputCommand);
    // var_dump($outputCommand);
}
else
    echo json_encode($output);


// TODO: should I delete the file?
/* unlink($baseFileName); */

?>
