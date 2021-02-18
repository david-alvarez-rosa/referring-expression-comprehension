<!DOCTYPE html>


<html lang="en">
  <head>
    <title>TODO</title>
    <meta charset="UTF-8" />
    <meta name="description" content="TODO" />
    <meta name="keywords" content="TODO" />
    <meta name="author" content="David Álvarez Rosa" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="canonical" href="TODO" />

    <!-- CSS files. -->
    <link rel="stylesheet" href="css/main.css" />
    <link rel="stylesheet" href="css/audio.css" />
    <link rel="stylesheet" href="css/bootstrap.min.css" />

    <!-- Javascript files. -->
    <script defer src="js/jquery-3.5.1.slim.min.js"></script>
    <script defer src="js/imgFileNames.js"></script>
    <script defer src="js/main.js"></script>
    <script defer src="js/audiodisplay.js"></script>
    <script defer src="js/recorderjs/recorder.js"></script>
    <script defer src="js/audio.js"></script>
    <script defer src="js/bootstrap.bundle.min.js"></script>
  </head>


  <body>
    <?php include 'header.html'; ?>

    <div id="audio">
      <div id="viz">
        <canvas id="analyser" width="1024" height="500"></canvas>
        <canvas id="wavedisplay" width="1024" height="500"></canvas>
      </div>
      <div id="controls">
        <img id="record" src="img/mic128.png" onclick="toggleRecording(this);">
      </div>
      <button onclick="stopAudio();">Stop audio</button>
    </div>


    <main role="main" class="container">
      <h1 class="mt-5">Web title</h1>
      <p>Hey there!</p>

      <section id="sec:img">
        <h2>Choose Image</h2>
        <p>
          Select one of these images by clicking on it.
          <button class="btn btn-secondary float-right"
                  onclick="populateGallery();"
                  data-toggle="tooltip"
                  title="Load more images">
            More images
          </button>
        </p>
        <div id="gallery"></div>

        <form class="form-group mt-5" onsubmit="return addImg();">
          <label for="img-url">or enter image web URL</label>
          <input id="img-url" placeholder="Image URL" />
          <input class="btn btn-primary" type="submit" />
        </form>

        <form onsubmit="return uploadImg();">
          <label for="img-local">or choose file locally from computer</label>
		      <input id="img-local" type="file" accept="image/*" />
          <input type="submit" />
        </form>
      </section>


      <section id="sec:results">
        <h2>Results</h2>
        <div id="img-selected-warn" class="alert alert-warning">
          Please, <a href="#sec:img">choose image</a>.
        </div>
        <img id="img-selected" alt="Selected image." />

        <div>
          <form onsubmit="return addReferringExpression();">
            <label for="referring-expression">
              Enter referring expression using your keyboard
            </label>
            <input id="referring-expression" placeholder="Referring expression" />
            <input type="submit" />
          </form>

          <p>or use microphone</p>
          <button type="button" onclick="startAudio();">Start audio</button>

          <h3>Selected referring expression</h3>
          <div id="re-selected-warn" class="alert alert-warning">
            Please, enter a referring expression.
          </div>
          <p id="re-selected"></p>
        </div>
      </section>
    </main>


    <?php include 'footer.html'; ?>


  </body>
</html>
