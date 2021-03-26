<?php
$fileName = 'audio/' . uniqid() . '.wav';

move_uploaded_file($_FILES["audio"]["tmp_name"], $fileName);

$command = '. Code/.venv/bin/activate 2>&1 &&'.
         ' XDG_CACHE_HOME=.cache/ TMP=.cache/' .
         ' python -W ignore Code/Prueba/main.py' .
         ' --file ' . $fileName . ' 2>&1';

exec($command, $outputCommand);

$output = [
    'command' => $command,
    'outputCommand' => $outputCommand,
];

if (isset($_POST['debug'])) {
    var_dump($output);
    print_r($outputCommand);
}
else
    echo json_encode($output);


?>
