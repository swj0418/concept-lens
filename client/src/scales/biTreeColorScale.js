import * as d3 from "d3"

export let biTreeColorScale = (chroma_range, lum_range=[40,97], depth) => {
    let luminance_scale = d3.scaleLinear().domain([depth,0]).range([lum_range[1],lum_range[0]])
    let chroma_scale = d3.scaleLinear().domain([depth,0]).range(chroma_range)
    return (elem) => d3.hcl(200,chroma_scale(elem),luminance_scale(elem))
}