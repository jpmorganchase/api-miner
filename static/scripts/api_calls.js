var query = document.getElementById("query")
var element = document.getElementById("results")

var jsonEditor = CodeMirror.fromTextArea(document.getElementById("api_query"), {
  matchBrackets: true
});


function codeReader(node, input_type) {
    CodeMirror.fromTextArea(node, {
    matchBrackets: true,
    mode: input_type,
    readOnly: true,
  })
};

function createErrorMsg(error) {
    node = document.createElement("div")
    node.id = "error_msg"
    error_html = error.replace(/\n/g, '<br>')
    node.innerHTML = 'Oops...Please bear with us! We found an error: <br> <br>' + error_html
    return node
}

function disableReqButton(state){
    
     var buttonElements = document.getElementsByClassName("request-button");
     var idx;
     for (idx = 0; idx < buttonElements.length; idx++){
        buttonElements[idx].disabled = state;
    }
}   

function setLoadingScreen(state){
  var loadingScreen = document.getElementById("loading");
  loadingScreen.style.display = state
}   

function search() {

  disableReqButton(true)
  setLoadingScreen("flex")

  element.innerHTML = '';
  jsonEditor.save()
  fetch('/search', {
      method: 'POST', 
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        'query_code': document.getElementById("api_query").value,
      })
    }
  )
  .then(response => {
    setLoadingScreen("none")
    if(!response.ok) {
        response.text().then(error => {
          // Format error message with linebreak
          node = createErrorMsg(error)
          element.appendChild(node)
          console.error(error)
          disableReqButton(false)
        })
        throw new Error()
      }
    else {
       return response.json();
     }
  })
  .then(data => {
      for(let i = 0; i < data.length; i++){
          node = document.createElement("div")
          element.appendChild(node)
          
          metadata = document.createElement("h4")

          let endpoint = data[i].endpoint
          let similarity = data[i].similarity

          metadata.innerHTML = `
              Endpoint: ${endpoint} <br/> 
              Similarity Score: ${similarity} <br/> 
          `
          metadata.style['padding-left'] = '20px'
          node.appendChild(metadata)

          codeNode = document.createElement("textarea")
          codeNode.innerHTML = JSON.stringify(data[i].spec_snippets, null, 2)
          node.appendChild(codeNode)  
          codeReader(codeNode, "application/json")
      }
      disableReqButton(false)
  })
  .catch(error => {
    console.log(error)
  })
}
