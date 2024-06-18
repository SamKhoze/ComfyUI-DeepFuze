import { $el } from "../../../scripts/ui.js";

function addStylesheet(url) {
	if (url.endsWith(".js")) {
		url = url.substr(0, url.length - 2) + "css";
	}
	$el("link", {
		parent: document.head,
		rel: "stylesheet",
		type: "text/css",
		href: url.startsWith("http") ? url : getUrl(url),
	});
}
function getUrl(path, baseUrl) {
	if (baseUrl) {
		return new URL(path, baseUrl).toString();
	} else {
		return new URL("../" + path, import.meta.url).toString();
	}
}

addStylesheet(getUrl("styles.css", import.meta.url));