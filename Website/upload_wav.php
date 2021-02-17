<?php
$tmp_filename = $_FILES["that_random_filename_wav"]["tmp_name"];


$tmp_filename = "audio.wav";    // TODO remove this


exec("scp audio.wav dalvarez@q.vectorinstitute.ai:~/testing/ 2>&1", $output);
var_dump($output);
echo exec("ssh dalvarez@q.vectorinstitute.ai 'testing/hola" . $tmp_filename . "'");
echo exec("ls");

move_uploaded_file($tmp_filename, "uploaded_audio.wav");
?>
