// Show selected image (from gallery, url or local storage).
function showSelectedImg(src, method) {
    let imgSelected = document.getElementById("img-selected");
    imgSelected.setAttribute("src", src);
    imgSelected.setAttribute("data-method", method);
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
    showSelectedImg(selectedImgSrc, "gallery");
}


// Add image via URL.
function addImg() {
    let imgUrl = document.getElementById("img-url").value;
    showSelectedImg(imgUrl, "url");
    return false; // Prevent form to be submitted.
}


// Upload image locally from computer.
function uploadImg() {
    let imgLocal = document.getElementById("img-local");
    let uploadedImg = imgLocal.files[0];

    const fileReader = new FileReader();
    fileReader.addEventListener("load", function () {
        showSelectedImg(this.result, "local");
    });
    fileReader.readAsDataURL(uploadedImg);

    return false; // Prevent form to be submitted.
}


// Checks if an image have been already selected.
function isImgSelected() {
    let imgSelected = document.getElementById("img-selected");
    if (imgSelected.src === "")
        return false;
    return true;
}


// Enter referring expression.
function addReferringExpression() {
    if (!isImgSelected()) {
        let imgSelectedWarn = document.getElementById("img-selected-warn");
        imgSelectedWarn.classList.remove("alert-warning");
        imgSelectedWarn.classList.add("alert-danger");
        imgSelectedWarn.scrollIntoView({ behavior: "smooth" });
        return false;
    }

    let referringExpression = document.getElementById("referring-expression");
    if (referringExpression.value === "")
        return false;
    let reSelected = document.getElementById("re-selected");
    reSelected.textContent = referringExpression.value;
    reSelected.style.display = "block";
    let reSelectedWarn = document.getElementById("re-selected-warn");
    reSelectedWarn.style.display = "none";

    // Execute code.
    segmentImg();

    return false; // Prevent form to be submitted.
}

function addReferringExpressionFromString(referringExpression) {
    let reContainer = document.getElementById("referring-expression");
    reContainer.value = referringExpression;
    addReferringExpression();
}



// Populate website gallery with random images from MSCOCO dataset.
function populateGallery() {
    const gallerySize = 12;
    let gallery = document.getElementById("gallery");
    gallery.innerHTML = "";
    let imgNumbers = [];
    for (let i = 0; i < gallerySize; ++i) {
        // Choose random number differnt from previous.
        let imgNumber = Math.round(Math.random()*(imgFileNames.length - 1));
        while (imgNumbers.includes(imgNumber))
            imgNumber = Math.round(Math.random()*(imgFileNames.length - 1));
        imgNumbers.push(imgNumber);

        // Set gallery image.
        let imgFileName = imgFileNames[imgNumber];
        let imgSrc = "datasets/refcoco/images/" + imgFileName;
        let galleryImg = document.createElement("img");
        galleryImg.setAttribute("src", imgSrc);
        galleryImg.setAttribute("alt", imgFileName);
        galleryImg.classList.add("rounded");
        galleryImg.setAttribute("data-toggle", "tooltip");
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


// Activate all tooltips.
$(function () {
    $('[data-toggle="tooltip"]').tooltip();
})


// Segment image with referring expression.
function segmentImg() {
    let imgSelected = document.getElementById("img-selected");
    let imgSrc = imgSelected.src;
    let reSelected = document.getElementById("re-selected");
    let referringExpression = reSelected.innerText;

    let formData = new FormData();
    formData.append("referringExpression", referringExpression);
    formData.append("imgMethod", imgSelected.dataset.method);
    formData.append("imgSrc", imgSrc);

    fetch("api/comprehend.php", {
        method: "POST",
        body: formData
    }).then(response => response.json())
        .then(response => {
            console.log(response);
            let img = document.getElementById("img-segmented");
            img.setAttribute("src", response['resultImgSrc']);
        });
}


// Toggle recording auxiliary function.
function toggleRecordingAux(event) {
    if (event.classList.contains("recording")) {
        // Start recording.
        event.title = "Stop recording";
    } else {
        // Stop recording.
        event.title = "Start recording";
        saveAudio();
        stopAudio();
    }
}


// Show warning message.
$(document).ready(function(){
        $("#warningModal").modal('show');
});
