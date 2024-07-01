import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
    node?.graph?.setDirtyCanvas(true);
}
function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r
        };
    } else {
        object[property] = callback;
    }
}

function addPreviewOptions(nodeType) {
    chainCallback(nodeType.prototype, "getExtraMenuOptions", function(_, options) {
        // The intended way of appending options is returning a list of extra options,
        // but this isn't used in widgetInputs.js and would require
        // less generalization of chainCallback
        let optNew = []
        try {
            const previewWidget = this.widgets.find((w) => w.name === "audiopreview");

            let url = null
            if (previewWidget.audioEl?.hidden == false && previewWidget.audioEl.src) {
                //Use full quality audio
                //url = api.apiURL('/view?' + new URLSearchParams(previewWidget.value.params));
                url = previewWidget.audioEl.src
            }
            if (url) {
                optNew.push(
                    {
                        content: "Open preview",
                        callback: () => {
                            window.open(url, "_blank")
                        },
                    },
                    {
                        content: "Save preview",
                        callback: () => {
                            const a = document.createElement("a");
                            a.href = url;
                            a.setAttribute("download", new URLSearchParams(previewWidget.value.params).get("filename"));
                            document.body.append(a);
                            a.click();
                            requestAnimationFrame(() => a.remove());
                        },
                    }
                );
            }
            if(options.length > 0 && options[0] != null && optNew.length > 0) {
                optNew.push(null);
            }
            options.unshift(...optNew);
            
        } catch (error) {
            console.log(error);
        }
        
    });
}
function previewAudio(node,file,type){
    var element = document.createElement("div");
    const previewNode = node;
    var previewWidget = node.addDOMWidget("audiopreview", "preview", element, {
        serialize: false,
        hideOnZoom: false,
        getValue() {
            return element.value;
        },
        setValue(v) {
            element.value = v;
        },
    });
    previewWidget.computeSize = function(width) {
        if (this.aspectRatio && !this.parentEl.hidden) {
            let height = (previewNode.size[0]-20)/ this.aspectRatio + 10;
            if (!(height > 0)) {
                height = 0;
            }
            this.computedHeight = height + 10;
            return [width, height];
        }
        return [width, -4];//no loaded src, widget should not display
    }
    // element.style['pointer-events'] = "none"
    previewWidget.value = {hidden: false, paused: false, params: {}}
    previewWidget.parentEl = document.createElement("div");
    previewWidget.parentEl.className = "audio_preview";
    previewWidget.parentEl.style['width'] = "100%"
    element.appendChild(previewWidget.parentEl);
    previewWidget.audioEl = document.createElement("audio");
    previewWidget.audioEl.controls = true;
    previewWidget.audioEl.loop = false;
    previewWidget.audioEl.muted = false;
    previewWidget.audioEl.style['width'] = "100%"
    previewWidget.audioEl.addEventListener("loadedmetadata", () => {

        previewWidget.aspectRatio = previewWidget.audioEl.audioWidth / previewWidget.audioEl.audioHeight;
        fitHeight(this);
    });
    previewWidget.audioEl.addEventListener("error", () => {
        //TODO: consider a way to properly notify the user why a preview isn't shown.
        previewWidget.parentEl.hidden = true;
        fitHeight(this);
    });

    let params =  {
        "filename": file,
        "type": type,
    }
    
    previewWidget.parentEl.hidden = previewWidget.value.hidden;
    previewWidget.audioEl.autoplay = !previewWidget.value.paused && !previewWidget.value.hidden;
    let target_width = 256
    if (element.style?.width) {
        //overscale to allow scrolling. Endpoint won't return higher than native
        target_width = element.style.width.slice(0,-2)*2;
    }
    if (!params.force_size || params.force_size.includes("?") || params.force_size == "Disabled") {
        params.force_size = target_width+"x?"
    } else {
        let size = params.force_size.split("x")
        let ar = parseInt(size[0])/parseInt(size[1])
        params.force_size = target_width+"x"+(target_width/ar)
    }
    
    previewWidget.audioEl.src = api.apiURL('/view?' + new URLSearchParams(params));

    previewWidget.audioEl.hidden = false;
    previewWidget.parentEl.appendChild(previewWidget.audioEl)
}

app.registerExtension({
	name: "UVR5.AudioPreviewer",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		
			nodeType.prototype.onExecuted = function (data) {
				previewAudio(this, data.audio[0], data.audio[1]);
			}
            addPreviewOptions(nodeType)
		
	}
});
