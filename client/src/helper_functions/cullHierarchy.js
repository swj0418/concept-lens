// export function cullHierarchyByName(node, target_names) {
//     // For non-balanced tree without showing full depth, I need to append all of the child node (names) in the target_names.
//     if(node.leaf) {
//         // If its name is in target_names, return [that it is in target_nodes, node itself]
//         return [target_names.indexOf(node.name) !== -1, node]
//     }
//
//     let retained_nodes = []
//     for(let c = 0; c < node.children.length; c++) {
//         let culled_result = cullHierarchyByName(node.children[c], target_names)
//         if(culled_result[0])
//             retained_nodes.push(culled_result[1])
//     }
//
//     let new_node = retained_nodes.length > 0 ? ({
//         leaf: node.leaf,
//         leaves: node.leaves,
//         name: node.name,
//         depth: node.depth,
//         quantity: node.quantity,
//         var: node.var,
//         children: retained_nodes
//     }) : null
//
//     return [retained_nodes.length > 0, new_node]
// }

export function cullHierarchyByNameAndQuantity(node, target_names, subsets) {
    // For non-balanced tree without showing full depth,
    // I need to append all the child node (names) in the target_names.
    if(node.leaf) {
        // If its name is in target_names, return [that it is in target_nodes, node itself]
        // let childrenNames = node.leaves.map(d => d.name)
        let childrenNames = node.leaf_indices
        const found = childrenNames.some(r => target_names.includes(r))
        let retained_leaves = []
        if (found) {
            for (var t in target_names) {
                for (var c in childrenNames) {
                    if (target_names[t] === childrenNames[c]) {
                        // console.log(node.leaves[c])
                        retained_leaves.push(node.leaves[c])
                    }

                }
            }
        }
        node.leaves = retained_leaves // Correct size.
        node.quantity = retained_leaves.length

        return [found, node]
    }

    let retained_nodes = []
    for(let c = 0; c < node.children.length; c++) {
        let culled_result = cullHierarchyByNameAndQuantity(node.children[c], target_names, subsets)
        if(culled_result[0])
            retained_nodes.push(culled_result[1])
    }

    let q = 0
    for (var n of retained_nodes) {
        // Their leaf indices, how many
        let names = n.leaves.map(d => d.name)
        q += names.filter(n => target_names.includes(n)).length
        // if (n.depth === 8) {
        //     console.log( n.depth, names.length, target_names.length, "   Quantity: ", q)
        // }
        // console.log( n.depth, names.length, target_names.length, "   Quantity: ", q)
    }

    let new_node = retained_nodes.length > 0 ? ({
        leaf: node.leaf,
        leaves: node.leaves,
        weaveScore: node.weaveScore,
        name: node.name,
        depth: node.depth,
        quantity: q,
        var: node.var,
        contributions: node.contributions,
        children: retained_nodes
    }) : null

    return [retained_nodes.length > 0, new_node]
}

export function cullHierarchyByNameTruncatedTree(node, target_names) {
    // For non-balanced tree without showing full depth,
    // I need to append all the child node (names) in the target_names.
    if(node.leaf) {
        // If its name is in target_names, return [that it is in target_nodes, node itself]
        // let childrenNames = node.leaves.map(d => d.name)
        let childrenNames = node.leaf_indices
        const found = childrenNames.some(r => target_names.includes(r))
        let retained_leaves = []
        if (found) {
            for (var t in target_names) {
                for (var c in childrenNames) {
                    if (target_names[t] === childrenNames[c])
                        retained_leaves.push(node.leaves[c])
                }
            }
        }
        node.leaves = retained_leaves

        return [found, node]
    }

    let retained_nodes = []
    for(let c = 0; c < node.children.length; c++) {
        let culled_result = cullHierarchyByNameTruncatedTree(node.children[c], target_names)
        if(culled_result[0])
            retained_nodes.push(culled_result[1])
    }

    let new_node = retained_nodes.length > 0 ? ({
        leaf: node.leaf,
        leaves: node.leaves,
        weaveScore: node.weaveScore,
        name: node.name,
        depth: node.depth,
        quantity: node.quantity,
        var: node.var,
        contributions: node.contributions,
        children: retained_nodes
    }) : null

    return [retained_nodes.length > 0, new_node]
}