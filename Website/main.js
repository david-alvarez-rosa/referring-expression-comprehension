function testing() {
    alert("hello world");
}


let gallery = document.getElementById("gallery");
let galleryImgs = gallery.getElementsByTagName("img");

for (galleryImg of galleryImgs)
    galleryImg.onclick = testing;
