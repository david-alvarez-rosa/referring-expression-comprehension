<?php
header('Content-Type:application/json');

$baseFileName = 'tmp/' . uniqid();

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
$command = 'source ../.venv/bin/activate &&'.
         ' XDG_CACHE_HOME=.cache/ MPLCONFIGDIR=.cache/' .
         ' python ../comprehend.py' .
         ' --resume ../checkpoints/model_refcoco.pth' .
         ' --img ' . $fileName .
         ' --sent "' . $referringExpression . '"' .
         ' --device cpu' .
         ' --output ' . $baseFileName . '.out.jpg 2>&1';


exec($command, $output);
// var_dump($output);


echo json_encode($baseFileName . '.out.jpg');


// TODO: should I delete the file?
/* unlink($baseFileName); */

?>
