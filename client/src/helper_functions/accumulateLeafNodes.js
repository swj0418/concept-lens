// export function accumulateLeafNodes(node, container) {
//     if (node.quantity === 1)
//         container.push(node)
//     else
//         node.children.map(d => accumulateLeafNodes(d, container))
// }
//
// export function accumulateLeafNodesBudget(node, container) {
//     if (node.children.length !== 0 && node.quantity === 1)
//         container.push(node)
//     else
//         node.children.map(d => accumulateLeafNodes(d, container))
// }

export function accumulateLeafNodes(node, container) {
    // if (node.quantity === 1)
    if (node.quantity <= 1)
        container.push(node)
    else
        node.children.map(d => accumulateLeafNodes(d, container))
}

// export function accumulateLeafNodesBudget(node, container) {
//     if (node.leaf === true && node.quantity === 1) {
//         container.push(node)
//     } else {
//         node.children.map(d => accumulateLeafNodesBudget(d, container))
//     }
// }

export function accumulateLeafNodesBudget(node) {
    if (node.leaf) {
        return node.leaves
    } else {
        let tmp = []
        node.children.map(d => tmp.push(...accumulateLeafNodesBudget(d)))
        return tmp
    }
}

export function accumulateVisLeafNodes(node) {
    /*
        Accumulates (truncated) nodes that will be visualized into an array.
     */
    if (node.leaf) {
        return node.leaves
    } else {
        let tmp = []
        node.children.map(d => tmp.push(...accumulateVisLeafNodes(d)))
        return tmp
    }
}