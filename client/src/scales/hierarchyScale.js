import {nonuniformBandScale} from "./nonuniformBandScale"

export let hierarchyScale = () => {
    let hierarchy = null;
    let range_values = null;
    let inner_padding = null;
    let outer_padding = null;
    let inner_step = null;
    let outer_step = null;
    let nested_band_scales = null;

    let setup = function() {
        if(hierarchy==null || range_values==null)
            return
        let full_range = range_values[1]-range_values[0]

        let create_band_scale_hierarchy = function(node_scale, next_range, node, depth=0) {
            if(node.leaf)
                return

            let domain = node.children.map(d => [d.name,d.quantity])
            node_scale.scale = nonuniformBandScale()
                .domain(domain)
                .range(next_range)
            if(inner_step || outer_step) {
                node_scale.scale.stepInner(inner_step ? inner_step : 0).stepOuter(outer_step ? outer_step : 0);
            }
            else {
                node_scale.scale.paddingInner(inner_padding ? inner_padding : 0).paddingOuter(outer_padding ? outer_padding : 0)
            }

            node_scale.depth = depth
            node_scale.children = []
            for(let i = 0; i < node.children.length; i++) {
                let child = node.children[i]
                let child_scale = {}
                if(child.leaf) {
                    node_scale.children.push(child_scale)
                    continue
                }
                let pos = node_scale.scale(child.name), size = node_scale.scale.bandwidth(child.name)
                create_band_scale_hierarchy(child_scale, [0,size], child, depth+1)
                node_scale.children.push(child_scale)
            }
        }

        nested_band_scales = {}
        create_band_scale_hierarchy(nested_band_scales, range_values, hierarchy)
    }

    let scale = function(root, add_dummy_paths=false,target_depth=1) {
        let position_hierarchy = {...root, position:range_values[0],
            absolute_position:range_values[0], size:range_values[1]-range_values[0]};
    // let traverse_scales = (node, node_scale, pos_hierarchy) => {
    //     let children_vals = node.children.map(d => d.name);
    //     pos_hierarchy.children = [];
    //     pos_hierarchy.leaf = false;
    //
    //     // For log sizing
    //     let scalesLog = [];
    //     let totalSize = 0;
    //
    //     // Calculate the log-scaled sizes and the total size for normalization
    //     for (let i = 0; i < node.children.length; i++) {
    //         const childValue = node_scale.scale.bandwidth(children_vals[i]);
    //         const logValue = Math.log2(childValue);
    //         scalesLog.push(logValue);
    //         totalSize += logValue;
    //     }
    //
    //     let accumulatedPosition = 0;
    //     for (let i = 0; i < node.children.length; i++) {
    //         let val = children_vals[i];
    //         let relativeSize = scalesLog[i] / totalSize * node_scale.scale.range()[1];
    //         let pos = accumulatedPosition;
    //         let abs_pos = pos + pos_hierarchy.absolute_position;
    //
    //         let child_hierarchy = {
    //             ...node.children[i],
    //             position: pos,
    //             absolute_position: abs_pos,
    //             size: relativeSize
    //         };
    //
    //         pos_hierarchy.children.push(child_hierarchy);
    //         accumulatedPosition += relativeSize;
    //
    //         if (!node.children[i].leaf) {
    //             traverse_scales(node.children[i], node_scale.children[i], child_hierarchy);
    //         }
    //     }
    // };

        let traverse_scales = (node,node_scale,pos_hierarchy) => {
            let children_vals = node.children.map(d => d.name);
            pos_hierarchy.children = [];
            pos_hierarchy.leaf = false;

            // For log sizing
            let scalesLog = []
            let totalSize = 0
            for(let i = 0; i < node.children.length; i++) {
                scalesLog.push(Math.log10(node_scale.scale.bandwidth(children_vals[i])))
                totalSize += node_scale.scale.bandwidth(children_vals[i])
            }
            let sum = 0
            scalesLog.map(d => {
                sum += d
            })
            let unit = totalSize / sum
            for(let i = 0; i < node.children.length; i++) {
                let val = children_vals[i]
                let pos = node_scale.scale(val)
                let abs_pos = pos+pos_hierarchy.absolute_position;
                let child_hierarchy = {...node.children[i], position:pos, absolute_position:abs_pos, size:node_scale.scale.bandwidth(val)};
                // console.log(pos, abs_pos)
                // let child_hierarchy = {...node.children[i], position:pos, absolute_position:abs_pos, size:unit*scalesLog[i]};
                pos_hierarchy.children.push(child_hierarchy)
                if(!node.children[i].leaf)
                    traverse_scales(node.children[i],node_scale.children[i],child_hierarchy)
            }
        }
        traverse_scales(root,nested_band_scales,position_hierarchy)

        if(add_dummy_paths) {
            let add_paths = (node) => {
                if(node.depth===target_depth)
                    return

                for(let c = 0; c < node.children.length; c++) {
                    let child_node = node.children[c]
                    if(child_node.leaf && child_node.depth !== target_depth) {
                        child_node.leaf = false;
                        let new_node = {...child_node};
                        new_node.leaf = true;
                        new_node.depth += 1;
                        new_node.position = 0;
                        child_node.children = [new_node];
                    }
                    add_paths(child_node);
                }
            }
            add_paths(position_hierarchy);
        }

        return position_hierarchy
    }

    scale.domain = function(hierarchy_data=null) {
        if(hierarchy_data===null)
            return hierarchy_data
        hierarchy = hierarchy_data
        setup()
        return this
    }

    scale.range = function(range_vals=null) {
        if(range_vals===null)
            return range_values
        range_values = range_vals
        setup()
        return this
    }

    scale.paddingInner = function(pad=null) {
        if(pad===null)
            return inner_padding
        inner_padding = pad
        setup()
        return this
    }

    scale.paddingOuter = function(pad=null) {
        if(pad===null)
            return outer_padding
        outer_padding = pad
        setup()
        return this
    }

    scale.stepInner = function(step=null) {
        if(step===null)
            return inner_step
        inner_step = step
        setup()
        return this
    }

    scale.stepOuter = function(step=null) {
        if(step===null)
            return outer_step
        outer_step = step
        setup()
        return this
    }

    return scale
}