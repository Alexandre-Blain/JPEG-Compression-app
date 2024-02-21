chrome.runtime.onInstalled.addListener(function() {
    console.log("Image Huffman Encoder extension installed.");
    // Perform some initialization or state setup if needed
});

// Listen for messages from the popup or content scripts
chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if(request.message === "uploadImage") {
            // Handle the image upload process here
        }
    }
);

