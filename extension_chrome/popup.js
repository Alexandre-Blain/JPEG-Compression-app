document.getElementById('uploadButton').addEventListener('click', function(){
    var file = document.getElementById('imageInput').files[0];
    var formData = new FormData();
    formData.append("image", file);

    // Replace with your Flask server's URL
    fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.text())
    .then(code => {

        var lastIndexOfT = code.lastIndexOf('t');
        var extractedSubstring = code.slice(lastIndexOfT + 1);
        var formattedSubstring = Number(extractedSubstring).toLocaleString();

        var lastCharacterDisplay = document.getElementById('lastCharacterDisplay');
        lastCharacterDisplay.textContent = 'Gain de place en octets: ' + formattedSubstring;


        var downloadLink = document.getElementById('downloadLinkTXT');
        downloadLink.href = 'data:text/plain;charset=utf-8,' + encodeURIComponent(code);
        downloadLink.download = 'huffman_code.txt';
        downloadLink.style.display = 'block';
    });
});
document.getElementById('decodeButton').addEventListener('click', function(){
    var file = document.getElementById('imageInput').files[0];
    // Prepare the file to be sent as form data
    var formData = new FormData();
    formData.append("file", file);

    // Replace with your Flask server's URL
    fetch('http://localhost:5000/decode', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())  // Handle the response as a blob
    .then(blob => {
        // Create an object URL for the blob
        var url = URL.createObjectURL(blob);
        // Create a download link for the image
        var downloadLink = document.getElementById('downloadLinkIMG'); // Ensure you have an element with id 'downloadLink'
        downloadLink.href = url;
        downloadLink.download = 'decoded_image.png';  // Set the suggested filename for the download
        downloadLink.style.display = 'block';  // Make the download link visible// Optionally auto-click the link to start download
    })
    .catch(error => {
        console.error('Error:', error);
    });
});