<?php
$tmp_filename = $_FILES["audio"]["tmp_name"];


move_uploaded_file($tmp_filename, "results/uploaded_audio.wav");


// exec("scp audio.wav dalvarez@q.vectorinstitute.ai:~/testing/ 2>&1", $output);
// var_dump($output);
// echo exec("ssh dalvarez@q.vectorinstitute.ai 'testing/hola" . $tmp_filename . "'");
// echo exec("ls");

// move_uploaded_file($tmp_filename, "uploaded_audio.wav");
?>
