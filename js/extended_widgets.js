//extended_widgets.js
import { api } from "/scripts/api.js"
import { ComfyWidgets } from "/scripts/widgets.js";

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

function addVideo(node, name,src, app,autoplay_value) {
	const MIN_SIZE = 50;
	
	function computeSize(size) {
		try{
	
		if (node.widgets[0].last_y == null) return;

		let y = node.widgets[0].last_y;
		let freeSpace = size[1] - y;

		// Compute the height of all non customvideo widgets
		let widgetHeight = 0;
		const multi = [];
		for (let i = 0; i < node.widgets.length; i++) {
			const w = node.widgets[i];
			if (w.type === "customvideo") {
				multi.push(w);
			} else {
				if (w.computeSize) {
					widgetHeight += w.computeSize()[1] + 4;
				} else {
					widgetHeight += LiteGraph.NODE_WIDGET_HEIGHT + 4;
				}
			}
		}
	
		// See how large each text input can be
		freeSpace -= widgetHeight;
		freeSpace /= multi.length + (!!node.imgs?.length);

		if (freeSpace < MIN_SIZE) {
			// There isnt enough space for all the widgets, increase the size of the node
			freeSpace = MIN_SIZE;
			node.size[1] = y + widgetHeight + freeSpace * (multi.length + (!!node.imgs?.length));
			node.graph.setDirtyCanvas(true);
		}

		// Position each of the widgets
		for (const w of node.widgets) {
			w.y = y;
			if (w.type === "customvideo") {
				y += freeSpace;
				w.computedHeight = freeSpace - multi.length*4;
			} else if (w.computeSize) {
				y += w.computeSize()[1] + 4;
			} else {
				y += LiteGraph.NODE_WIDGET_HEIGHT + 4;
			}
		}

		node.inputHeight = freeSpace;
	}catch(e){
		
	}
	}
	const widget = {
		type: "customvideo",
		name,
		get value() {
			return this.inputEl.value;
		},
		set value(x) {
			this.inputEl.value = x;
		},
		draw: function (ctx, _, widgetWidth, y, widgetHeight) {
			if (!this.parent.inputHeight) {
				// If we are initially offscreen when created we wont have received a resize event
				// Calculate it here instead
				node.setSizeForImage?.();
				
			}
			const visible = app.canvas.ds.scale > 0.5 && this.type === "customvideo";
			const margin = 10;
			const elRect = ctx.canvas.getBoundingClientRect();
			const transform = new DOMMatrix()
				.scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
				.multiplySelf(ctx.getTransform())
				.translateSelf(margin, margin + y);

			const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
			Object.assign(this.inputEl.style, {
				transformOrigin: "0 0",
				transform: scale,
				left: `${transform.a + transform.e}px`,
				top: `${transform.d + transform.f}px`,
				width: `${widgetWidth - (margin * 2)}px`,
				height: `${this.parent.inputHeight - (margin * 2)}px`,
				position: "absolute",
				background: (!node.color)?'':node.color,
				color: (!node.color)?'':'white',
				zIndex: app.graph._nodes.indexOf(node),
			});
			this.inputEl.hidden = !visible;
		},
	};


	widget.inputEl = document.createElement("video");

	
	// Set the video attributes
	Object.assign(widget.inputEl, {
		controls: true,
		src: src,
		poster: "",
		width: 400,
		height: 300,
		loop: true,
		muted: true,
		autoplay: autoplay_value,
		type : "video/mp4"
		
	});
	//widget.inputEl.classList.add("dididi");


	
	// Add video element to the body
	document.body.appendChild(widget.inputEl);



	widget.parent = node;
	document.body.appendChild(widget.inputEl);

	node.addCustomWidget(widget);

	app.canvas.onDrawBackground = function () {
		// Draw node isnt fired once the node is off the screen
		// if it goes off screen quickly, the input may not be removed
		// this shifts it off screen so it can be moved back if the node is visible.
		for (let n in app.graph._nodes) {
			n = graph._nodes[n];
			for (let w in n.widgets) {
				let wid = n.widgets[w];
				if (Object.hasOwn(wid, "inputEl")) {
					wid.inputEl.style.left = -8000 + "px";
					wid.inputEl.style.position = "absolute";
				}
			}
		}
	};

	node.onRemoved = function () {
		// When removing this node we need to remove the input from the DOM
		for (let y in this.widgets) {
			if (this.widgets[y].inputEl) {
				this.widgets[y].inputEl.remove();
			}
		}
	};

	widget.onRemove = () => {
		widget.inputEl?.remove();

		// Restore original size handler if we are the last
		if (!--node[MultilineSymbol]) {
			node.onResize = node[MultilineResizeSymbol];
			delete node[MultilineSymbol];
			delete node[MultilineResizeSymbol];
		}
	};

	if (node[MultilineSymbol]) {
		node[MultilineSymbol]++;
	} else {
		node[MultilineSymbol] = 1;
		const onResize = (node[MultilineResizeSymbol] = node.onResize);

		node.onResize = function (size) {
	
			computeSize(size);
			// Call original resizer handler
			if (onResize) {
				onResize.apply(this, arguments);
			}
		};
	}

	return { minWidth: 400, minHeight: 200, widget };
}


export function showVideoInput(name,node) {
	const videoWidget = node.widgets.find((w) => w.name === "videoWidget");
	const temp_web_url = node.widgets.find((w) => w.name === "local_url");
	
	
	let folder_separator = name.lastIndexOf("/");
	let subfolder = "n-suite";
	if (folder_separator > -1) {
		subfolder = name.substring(0, folder_separator);
		name = name.substring(folder_separator + 1);
	}

	let url_video = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=input&subfolder=${subfolder}${app.getPreviewFormatParam()}`);
	videoWidget.inputEl.src = url_video
	temp_web_url.value = url_video
}

export function showVideoOutput(name,node) {
	const videoWidget = node.widgets.find((w) => w.name === "videoOutWidget");

	
	
	let folder_separator = name.lastIndexOf("/");
	let subfolder = "n-suite/videos";
	if (folder_separator > -1) {
		subfolder = name.substring(0, folder_separator);
		name = name.substring(folder_separator + 1);
	}


	let url_video = api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=output&subfolder=${subfolder}${app.getPreviewFormatParam()}`);
	videoWidget.inputEl.src = url_video

	return url_video;
}



export const ExtendedComfyWidgets = {
    ...ComfyWidgets, // Copy all the functions from ComfyWidgets
	
	VIDEO(node, inputName, inputData, src, app,type="input",autoplay_value=true) {
	try {	
		const videoWidget = node.widgets.find((w) => w.name === "video");
		const autoplay = node.widgets.find((w) => w.name === "autoplay");
		const defaultVal = "";
		let res;
		res = addVideo(node, inputName, src, app,autoplay_value);
		
		if (type == "input"){

			const cb = node.callback;
			videoWidget.callback = function () {
				
				showVideoInput(videoWidget.value, node);
				if (cb) {
					return cb.apply(this, arguments);
				}
			};
			autoplay.callback = function () {
				const videoWidgetz = node.widgets.find((w) => w.name === "videoWidget");
			
				videoWidgetz.inputEl.autoplay = autoplay.value;
				showVideoInput(videoWidget.value, node);
				if (cb) {
					return cb.apply(this, arguments);
				}
			}
		}

		if (node.type =="LoadVideoAdvanced"){
	

	}

		return res;	
	}
	catch (error) {

		console.error("Errore in extended_widgets.js:", error);
		throw error; 
	
	}

},


};
