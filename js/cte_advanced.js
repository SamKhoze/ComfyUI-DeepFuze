import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js"
const MultilineSymbol = Symbol();
const MultilineResizeSymbol = Symbol();

function getStyles(name) {
	//console.log("getStyles called " + name);
	
	return api.fetchApi('/nsuite/styles')
	  .then(response => response.json())
	  .then(data => {
		// Eseguire l'elaborazione dei dati
		const styles = data.styles;
		//console.log('Styles:', styles);
		let positive_prompt = "";
		let negative_prompt = "";
  
		// Funzione per ottenere positive_prompt e negative_prompt dato il name
		for (let i = 0; i < styles[0].length; i++) {
		  const style = styles[0][i];
		  if (style.name === name) {
			positive_prompt = style.prompt;
			negative_prompt = style.negative_prompt;
			//console.log('Style:', style.name);
			break;
		  }
		}
  
		if (positive_prompt !== "") {
		  //console.log("Positive prompt:", positive_prompt);
		  //console.log("Negative prompt:", negative_prompt);
		  return { positive_prompt: positive_prompt, negative_prompt: negative_prompt };
		} else {
		  return { positive_prompt: "", negative_prompt: "" };
		}
	  })
	  .catch(error => {
		console.error('Error:', error);
		throw error; // Rilancia l'errore per consentire al chiamante di gestirlo
	  });
  }

  function addStyles(name, positive_prompt, negative_prompt) {
	  return api.fetchApi('/nsuite/styles/add', {
		method: 'POST',
		headers: {
		  'Content-Type': 'application/json',
		},
		body: JSON.stringify({
		  name: name,
		  positive_prompt: positive_prompt,
		  negative_prompt: negative_prompt
		}),
		  
	  })
  }

  function updateStyles(name, positive_prompt, negative_prompt) {
	return api.fetchApi('/nsuite/styles/update', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({
			name: name,
			positive_prompt: positive_prompt,
			negative_prompt: negative_prompt
		}),
	})
}

function removeStyles(name) {
	//confirmation
	let ok = confirm("Are you sure you want to remove this style?");
	if (!ok) {
		return;
	}

	return api.fetchApi('/nsuite/styles/remove', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({
			name: name
		}),
	})
}

app.registerExtension({
	name: "n.CLIPTextEncodeAdvancedNSuite",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

		const onAdded = nodeType.prototype.onAdded;
		if (nodeData.name === "CLIPTextEncodeAdvancedNSuite [n-suite]") {
		nodeType.prototype.onAdded = function () {
			onAdded?.apply(this, arguments);
			const styles = this.widgets.find((w) => w.name === "styles");
			const p_prompt  = this.widgets.find((w) => w.name === "positive_prompt");
			const n_prompt = this.widgets.find((w) => w.name === "negative_prompt");
			const cb = nodeData.callback;
			let addedd_positive_prompt = "";
			let addedd_negative_prompt = "";
			styles.callback = function () {
				let index = styles.options.values.indexOf(styles.value); 
				 
		
				if (addedd_positive_prompt == "" && addedd_negative_prompt == "") {
					getStyles(styles.options.values[index-1]).then(style_prompts => {
						//wait 4 seconds
					
						console.log(style_prompts);
				
						addedd_positive_prompt =  style_prompts.positive_prompt;
						addedd_negative_prompt =  style_prompts.negative_prompt;
						//alert("Addedd positive prompt: " + addedd_positive_prompt + "\nAddedd negative prompt: " + addedd_negative_prompt);
					})
				}


				let current_positive_prompt = p_prompt.value;
				let current_negative_prompt = n_prompt.value;
				
				getStyles(styles.value).then(style_prompts => {
					//console.log(style_prompts)

					if ((current_positive_prompt.trim() != addedd_positive_prompt.trim() || current_negative_prompt.trim() != addedd_negative_prompt.trim())) {
						
						let ok = confirm("Style has been changed. Do you want to change style without saving?");

						
						if (!ok) {
							if (styles.value === styles.options.values[0]) {
								value = styles.options.values[0];
							}
							styles.value = styles.options.values[index-1];

							
							return;
						}
					}

					// add the addedd prompt to the current prompt
					p_prompt.value =  style_prompts.positive_prompt;
					n_prompt.value =  style_prompts.negative_prompt;


					addedd_positive_prompt = style_prompts.positive_prompt;
					addedd_negative_prompt = style_prompts.negative_prompt;
					if (cb) {
						return cb.apply(this, arguments);
					}
				  })
				  .catch(error => {
					console.error('Error:', error);
				  });
				
			};

			

			let savestyle;
			let replacestyle;
			let deletestyle;
		
			
			// Create the button widget for selecting the files
			savestyle = this.addWidget("button", "New", "image", () => {
				////console.log("Save called");
				//ask input name style
				let inputName = prompt("Enter a name for the style:", styles.value);
				if (inputName === null) {
					return;
				}
			
				
				addStyles(inputName, p_prompt.value, n_prompt.value);
				// Add the file to the dropdown list and update the widget value
				
				if (!styles.options.values.includes(inputName)) {
					styles.options.values.push(inputName);
				  }

			},{
				cursor: "grab",
			},);
			replacestyle = this.addWidget("button", "Replace", "image", () => {
				//console.log("Replace called");
				updateStyles(styles.value, p_prompt.value, n_prompt.value);
			},{
				cursor: "grab",
			},);
			deletestyle = this.addWidget("button", "Delete", "image", () => {
				//console.log("Delete called");
				removeStyles(styles.value);

				// Remove the file from the dropdown list
				styles.options.values = styles.options.values.filter((value) => value !== styles.value);
			},{
				cursor: "grab",
			},);
			savestyle.serialize = false;	
		
		}
	

		


		};
		
	},
});
