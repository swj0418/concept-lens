import * as d3 from "d3";

export function gatherWeaveScores(node, container) {
    if(node.leaf)
        container.push(node.weaveScore)
    else {
        node.children.map(d => gatherWeaveScores(d, container))
    }
}
