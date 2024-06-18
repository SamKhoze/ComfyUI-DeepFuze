import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js"
import { ExtendedComfyWidgets,showVideoOutput } from "./extended_widgets.js";
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
			//	showVideo(path,node);
				videoWidget.value = path;
				if (data.subfolder) path = data.subfolder + "/" + path;
				showVideo(path,node);
				
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
	name: "Comfy.VideoSave",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

		const onExecuted = nodeType.prototype.onExecuted;
	
			
		const onAdded = nodeType.prototype.onAdded;
		if (nodeData.name === "SaveVideo [n-suite]") {
		nodeType.prototype.onAdded = function () {

			ExtendedComfyWidgets["VIDEO"](this, "videoOutWidget", ["STRING"], "", app,"output");
		
		};
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);
			console.log(nodeData)

		let full_path="";

		for (const list of message.text) {
			full_path = list;
		}

		let fullweb= showVideoOutput(full_path,this)
		
}
		};
		
	},
});
