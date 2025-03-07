import * as d3 from "d3"

export let divergingDistributionColorScale = (hues,saturation,darkness,mean,std,n_quantize=null,m_std_interval=1.,u_std_interval=1.) => {
    // scheme: prioritize variance, so use hue to better distinguish different levels of variance
    // and then use a diverging color scheme to sweep through levels of mean
    let chroma_scale = d3.scaleLinear()
        .domain([mean-m_std_interval*2*std,m_std_interval*mean,mean+m_std_interval*2*std])
        // .domain([0, u_std_interval*std, u_std_interval*2*std])
        .range([saturation,0,saturation]).clamp(true)

    let lum_scale = d3.scaleLinear()
        .domain([mean-m_std_interval*2*std,m_std_interval*mean,mean+m_std_interval*2*std])
        .range([darkness,92,darkness]).clamp(true)

    let hue_left_scale = d3.scaleLinear()
        .domain([0,2*std])
        .range([hues[0],hues[1]]).clamp(true)
    let hue_right_scale = d3.scaleLinear()
        .domain([0,2*std])
        .range([hues[2],hues[3]]).clamp(true)

    let the_scale = (mean_val,std_val) => {
        return d3.hcl((mean_val < mean ? hue_left_scale(std_val) : hue_right_scale(std_val)), chroma_scale(mean_val), lum_scale(mean_val))
    }

    the_scale.mean = mean;
    the_scale.std = std;

    return the_scale
}