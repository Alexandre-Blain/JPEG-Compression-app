// Content script to interact with web pages

// Example: Listen for clicks on images
document.addEventListener('click', function(event) {
    var element = event.target;
    if(element.tagName === 'IMG'){
        console.log("Image clicked: ", element.src);
        // Optionally, send this information back to the popup or background script
        chrome.runtime.sendMessage({message: "imageClicked", url: element.src});
    }
});


