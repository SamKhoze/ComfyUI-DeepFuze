import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js"
import { ExtendedComfyWidgets,showVideoInput } from "./extended_widgets.js";
const MultilineSymbol = Symbol();
const MultilineResizeSymbol = Symbol();


async function uploadFile(file, updateNode, node, pasted = false) {
	const videoWidget = node.widgets.find((w) => w.name === "video");
	
	
	try {
		// Wrap file in formdata so it includes filename
		const body = new FormData();
		body.append("image", file);
		if (pasted) {
			body.append("subfolder", "pasted");
		}
		else {
			body.append("subfolder", "n-suite");
		}
	
		const resp = await api.fetchApi("/upload/image", {
			method: "POST",
			body,
		});

		if (resp.status === 200) {
			const data = await resp.json();
			// Add the file to the dropdown list and update the widget value
			let path = data.name;
			

			if (!videoWidget.options.values.includes(path)) {
				videoWidget.options.values.push(path);
			}
			
			if (updateNode) {
		
				videoWidget.value = path;
				if (data.subfolder) path = data.subfolder + "/" + path;
				showVideoInput(path,node);
				
			}
		} else {
			alert(resp.status + " - " + resp.statusText);
		}
	} catch (error) {
		alert(error);
	}
}




let uploadWidget = "";
app.registerExtension({
	name: "Comfy.VideoLoadAdvanced",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

		const onAdded = nodeType.prototype.onAdded;
		if (nodeData.name === "LoadVideo [n-suite]") {
		nodeType.prototype.onAdded = function () {
			onAdded?.apply(this, arguments);
			const temp_web_url = this.widgets.find((w) => w.name === "local_url");
			const autoplay_value = this.widgets.find((w) => w.name === "autoplay");
		
			
			let uploadWidget;
			const fileInput = document.createElement("input");
			Object.assign(fileInput, {
				type: "file",
				accept: "video/mp4,image/gif,video/webm",
				style: "display: none",
				onchange: async () => {
					if (fileInput.files.length) {
						await uploadFile(fileInput.files[0], true,this);
					}
				},
			});
			document.body.append(fileInput);
			// Create the button widget for selecting the files
			uploadWidget = this.addWidget("button", "choose file to upload", "image", () => {
				fileInput.click();
			},{
				cursor: "grab",
			},);
			uploadWidget.serialize = false;


		setTimeout(() => {
			ExtendedComfyWidgets["VIDEO"](this, "videoWidget", ["STRING"], temp_web_url.value, app,"input", autoplay_value.value);
		
		}, 100); 
		
		
		}
	

			nodeType.prototype.onDragOver = function (e) {
				if (e.dataTransfer && e.dataTransfer.items) {
					const image = [...e.dataTransfer.items].find((f) => f.kind === "file");
					return !!image;
				}
	
				return false;
			};
	
			// On drop upload files
			nodeType.prototype.onDragDrop = function (e) {
				console.log("onDragDrop called");
				let handled = false;
				for (const file of e.dataTransfer.files) {
					if (file.type.startsWith("video/mp4")) {
						
						const filePath = file.path || (file.webkitRelativePath || '').split('/').slice(1).join('/'); 


						uploadFile(file, !handled,this ); // Dont await these, any order is fine, only update on first one

						handled = true;
					}
				}
	
				return handled;
			};
	
			nodeType.prototype.pasteFile = function(file) {
				if (file.type.startsWith("video/mp4")) {

					//uploadFile(file, true, is_pasted);

					return true;
				}
				return false;
			}


		};
		
	},
});
