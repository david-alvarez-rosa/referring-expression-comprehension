// Show selected image (from gallery, url or local storage).
function showSelectedImg(src) {
    let imgSelected = document.getElementById("img-selected");
    imgSelected.setAttribute("src", src);
    imgSelected.style.display = "block";
    let imgSelectedWarn = document.getElementById("img-selected-warn");
    imgSelectedWarn.style.display = "none";
    // Scroll to results section.
    let resultsSection = document.getElementById("sec:results")
    resultsSection.scrollIntoView({ behavior: "smooth" });
}


// Select image from website gallery.
function selectImg(event) {
    let selectedImgSrc = event.target.src;
    showSelectedImg(selectedImgSrc);
}


// Add image via URL.
function addImg() {
    let imgUrl = document.getElementById("img-url").value;
    console.log(imgUrl);
    showSelectedImg(imgUrl);
    return false; // Prevent form to be submitted.
}


// Upload image locally from computer.
function uploadImg() {
    let imgLocal = document.getElementById("img-local");
    let uploadedImg = imgLocal.files[0];

    const fileReader = new FileReader();
    fileReader.addEventListener("load", function () {
        showSelectedImg(this.result);
    });
    fileReader.readAsDataURL(uploadedImg);

    return false; // Prevent form to be submitted.
}


// Enter referring expression.
function addReferringExpression() {
    let referringExpression = document.getElementById("referring-expression");
    if (referringExpression.value === "")
        return false;
    let reSelected = document.getElementById("re-selected");
    reSelected.textContent = referringExpression.value;
    reSelected.style.display = "block";
    let reSelectedWarn = document.getElementById("re-selected-warn");
    reSelectedWarn.style.display = "none";

    return false; // Prevent form to be submitted.
}


// Populate website gallery with random images from MSCOCO dataset.
function populateGallery() {
    const gallerySize = 10;
    let gallery = document.getElementById("gallery");
    gallery.innerHTML = "";
    for (let i = 0; i < gallerySize; ++i) {
        let imgNumber = Math.round(Math.random()*(imgFileNames.length - 1));
        let imgFileName = imgFileNames[imgNumber];
        let imgSrc = "datasets/refcoco/images/" + imgFileName;
        let galleryImg = document.createElement("img");
        galleryImg.setAttribute("src", imgSrc);
        galleryImg.setAttribute("alt", imgFileName);
        galleryImg.classList.add("rounded");
        galleryImg.setAttribute("title", "Select image " + imgFileName);
        galleryImg.onclick = selectImg;
        gallery.appendChild(galleryImg);
    }
}

populateGallery();


// Start audio recording.
function startAudio() {
    initAudio();
    let audioContainer = document.getElementById("audio");
    audioContainer.style.display = "block";
}
// window.addEventListener("load", startAudio);

// Stop audio recording.
function stopAudio() {
    let audioContainer = document.getElementById("audio");
    audioContainer.style.display = "none";
}
