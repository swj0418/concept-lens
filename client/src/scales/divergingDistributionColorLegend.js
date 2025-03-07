import * as d3 from "d3"

export let divergingDistributionColorLegend = (color_scale, parent_g, size, steps=4) => {
    let legend_g = parent_g.append('g')
    let mean = color_scale.mean, std = color_scale.std;
    let min_mean = mean-2*std, max_mean = mean+2*std, mean_range = (max_mean-min_mean);
    let min_std = 0, max_std = 2*std, std_range = 2*std;

    let background_color = d3.hcl(0,0,100);

    let x_scale = d3.scaleBand()
        .domain(d3.range(steps))
        .range([0,size])
        .paddingInner(.05)
    let y_scale = d3.scaleBand()
        .domain(d3.range(steps))
        .range([size,0])
        .paddingInner(.05)

    legend_g.append('rect')
        .attr('width', size).attr('height', size).attr('fill', background_color).attr('stroke', 'none')

    legend_g.selectAll()
        .data(d3.cross(d3.range(steps),d3.range(steps)))
        .join('rect')
        .attr('x', d => x_scale(d[0]))
        .attr('y', d => y_scale(d[1]))
        .attr('width', x_scale.bandwidth())
        .attr('height', y_scale.bandwidth())
        .attr('fill', d => {
            let mean_val = min_mean+.5*mean_range/(steps)+d[0]*mean_range/steps;
            let std_val = min_std+.5*std_range/(steps)+d[1]*std_range/steps;
            return color_scale(mean_val,std_val);
        })

    let gaussian = (v,m,s) => Math.exp(-((v-m)**2)/(s**2))
    let gaussian_func = d3.range(200).map(d => [3*(d/199-.5),gaussian(3*(d/199-.5),0,1)])

    let gaussian_x_scale = d3.scaleLinear().domain([-1.5,1.5]).range([0,size])
    let gaussian_y_scale = d3.scaleLinear().domain([0,1]).range([size/5,0])

    let area_mark = d3.area().x(d => gaussian_x_scale(d[0])).y0(size/5).y1(d => gaussian_y_scale(d[1]))
    let line_mark = d3.line().x(d => gaussian_x_scale(d[0])).y(d => gaussian_y_scale(d[1]))

    legend_g.append('g').attr('transform', `translate(0,${-size/5})`).append('path')
        .attr('fill', d3.hcl(0,0,90)).attr('stroke', 'none')//.attr('stroke', d3.hcl(0,0,8)).attr('stroke-width', .8)
        .attr('d', area_mark(gaussian_func))

    legend_g.selectAll()
        .data(d3.range(steps-1))
        .join('line')
        .attr('x1', d => x_scale(d+1)-.5*x_scale.step()*x_scale.paddingInner())
        .attr('x2', d => x_scale(d+1)-.5*x_scale.step()*x_scale.paddingInner())
        .attr('y1', 0)
        .attr('y2', d => {
            let normalized_coord = 3*((d+1)/(steps)-.5);
            return gaussian_y_scale(gaussian(normalized_coord,0,1))-size/5;
        })
        .attr('stroke', background_color).attr('stroke-width', 1.5)

    legend_g.append('g').attr('transform', `translate(0,${-size/5})`).append('path')
        .attr('stroke', d3.hcl(0,0,8)).attr('stroke-width', .8).attr('fill', 'none')
        .attr('d', line_mark(gaussian_func))

    legend_g.append('defs').append('marker')
        .attr('id', 'arrow')
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 5).attr('refY', 5)
        .attr('markerWidth', 6).attr('markerHeight', 6)
        .attr('orient', 'auto-start-reverse')
        .append('path').attr('d', 'M 0 0 L 10 5 L 0 10 z')

    let delta = 15;
    legend_g.append('line')
        .attr('x1', 0).attr('x2', size+delta)
        .attr('y1', size).attr('y2', size)
        .attr('stroke', d3.hcl(0,0,30)).attr('stroke-width', 1.5)
        .style('marker-end', 'url(#arrow)')

    legend_g.append('line')
        .attr('x1', 0).attr('x2', 0)
        .attr('y2', -delta).attr('y1', size)
        .attr('stroke', d3.hcl(0,0,30)).attr('stroke-width', 1.5)
        .style('marker-end', 'url(#arrow)')

    legend_g.append('g').attr('transform', `translate(${size - 20},${size+14})`)
        .append('text')
        .text('mag')
        .attr('font-size', '16px')
        .attr('font-weight', 'bold')
        .attr('fill', d3.hcl(0,0,35))

    legend_g.append('g').attr('transform', `translate(${-30},${0})`)
        .append('text')
        .text('std')
        .attr('font-size', '16px')
        .attr('font-weight', 'bold')
        .attr('fill', d3.hcl(0,0,35))
}