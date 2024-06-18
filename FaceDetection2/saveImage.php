<?php
$image = $_POST['image'];
$fileName = $_POST['fileName'];
$image = str_replace('data:image/png;base64,', '', $image);
$image = str_replace(' ', '+', $image);
$data = base64_decode($image);
$file = 'images/' . $fileName;
$success = file_put_contents($file, $data);
if ($success) {
    echo 'Image saved successfully';
} else {
    echo 'Error saving image';
}
?>