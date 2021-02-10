<?php
$tmp_filename = $_FILES["that_random_filename_wav"]["tmp_name"];
move_uploaded_file($tmp_filename, "uploaded_audio.wav");

echo exec("python main.py");
?>
