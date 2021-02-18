<?php
header('Content-Type:application/json');

/* json_encode($_POST); */
/* json_encode($_POST['imgSrc']); */
/* return; */

$fileName = 'tmp/' . uniqid();

switch ($_POST['imgMethod']) {
    case 'gallery':
        copy($_POST['imgSrc'], $fileName);
        break;
    case 'url':
        file_put_contents($fileName, file_get_contents($_POST['imgSrc']));
        break;
    case 'local':
        file_put_contents($fileName, base64_decode(
            explode(";base64,", $_POST['imgSrc'])[1])
        );
        break;
}

$referringExpression = $_POST['referringExpression'];

echo json_encode(exec('python ../echoHola.py'));


/* unlink($fileName); */


?>
