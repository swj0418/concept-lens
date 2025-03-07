import * as d3 from "d3"

// export function accumulateQuantity(node) {
//     if(node.quantity === 1)
//         return 1
//     let child_quantities = node.children.map(d => accumulateQuantity(d))
//     node.quantity = d3.sum(child_quantities)
//     return node.quantity
// }

export function accumulateQuantity(node) {
    if(node.leaf)
        return node.quantity
    else {
        let child_quantities = node.children.map(d => accumulateQuantity(d))
        node.quantity = d3.sum(child_quantities)
        return node.quantity
    }
}
