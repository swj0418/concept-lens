import * as d3 from "d3"

export let nonuniformBandScale = () => {
    let domain_elements = null;
    let range_values = null;
    let inner_padding = null;
    let outer_padding = null;
    let inner_step = null;
    let outer_step = null;
    let bws = null;
    let steps = null;

    let setup = function() {
        if(domain_elements==null || range_values==null)
            return

        let keys = domain_elements.map(d => d[0]), vals = domain_elements.map(d => d[1]);
        let V = d3.sum(vals);
        let full_range = range_values[1]-range_values[0];

        if(inner_step || outer_step) {
            let p_inner = inner_step || 0;
            let p_outer = outer_step || 0;

            let W_leftover = full_range - (2*p_outer + (vals.length-1)*p_inner);
            if(W_leftover < 0) {
                p_outer = 0;
                p_inner = 0;
                W_leftover = full_range;
            }
            bws = {};
            steps = {};
            let step = range_values[0] + p_outer;
            for(let i = 0; i < domain_elements.length; i++) {
                let key = keys[i];
                steps[key] = step;
                bws[key] = W_leftover*(vals[i]/V);
                step += bws[key]+p_inner;
            }
        }
        else {
            let p_inner = inner_padding || 0;
            let p_outer = outer_padding || 0;

            let W = (full_range*V)/(V + (p_inner/(1-p_inner))*d3.sum(vals.slice(0,-1)) + ((p_outer)/(1-p_inner))*(vals[0]+vals[vals.length-1]))

            bws = {}
            steps = {}
            let step = p_outer*(W*vals[0]/V)/(1-p_inner)
            for(let i = 0; i < domain_elements.length; i++) {
                let key = keys[i]
                steps[key] = step
                let next_step = (W*vals[i])/(V*(1-p_inner))
                bws[key] = next_step*(1-p_inner)
                step += next_step
            }
        }
    }

    let scale = function(elem) {
        return steps[elem]
    }

    scale.domain = function(domain_elems=null) {
        if(domain_elems===null)
            return domain_elements
        domain_elements = domain_elems
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

    scale.bandwidth = function(elem) {
        return bws[elem]
    }

    return scale
}