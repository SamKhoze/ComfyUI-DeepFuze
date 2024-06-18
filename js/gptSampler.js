import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
	name: "n.GPTSampler",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		
		if (nodeData.name === "GPT Sampler [n-suite]") {
			console.warn("GPTSampler detected")
			
			const onExecuted = nodeType.prototype.onExecuted;
	
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);
				
				const pos_cached = this.widgets.findIndex((w) => w.name === "cached");
				console.warn("value:"+pos_cached)
				
				if (this.widgets) {
					const pos_text = this.widgets.findIndex((w) => w.name === "text");
					if (pos_text !== -1) {
						for (let i = pos_text; i < this.widgets.length; i++) {
							this.widgets[i].onRemove?.();
						}
						this.widgets.length = pos_text;
					}
				}


				if (this.widgets[pos_cached].value === "NO") {
					
					const w = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app);
					//random seed
					var rnm = Math.floor(Math.random() * 10000)
					w.widget.value = rnm;
					
				
				}
			
			};
		}
	},
});
