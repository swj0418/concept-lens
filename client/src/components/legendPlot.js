import React, {Component} from "react";
import * as d3 from "d3"


function simpleHeatmap(data, m_scale, m_size, m_id, m_x, m_y) {
    let x = m_x ? m_x : 0;
    let y = m_y ? m_y : 0;
    let size = m_size ? m_size : 0;
    let scale = m_scale ? m_scale : () => "#fff";
    let id = m_id;
    let h;

    function heatmap(nel) {
        heatmap.el = nel;
        heatmap.setProperties();
    }

    heatmap.setProperties = function () {
        if (!this.el) {
            return;
        }

        if (!heatmap.svgGroup) {
            heatmap.svgGroup = heatmap.el.append("g");
        }

        heatmap.svgGroup.attr("transform", `translate(${x},${y})`);

        heatmap.svgGroup
            .selectAll("g")
            .data(data)
            .enter()
            .append("g")
            .selectAll("rect")
            .data((d, i) =>
                d.map((val) => ({
                    r: i,
                    v: val,
                }))
            )
            .enter()
            .append("rect")
            .datum(function (d, i) {
                d.c = i;
                return d;
            });

        heatmap.svgGroup
            .selectAll("g")
            .selectAll("rect")
            .attr("x", (d) => (size / data[d.r].length) * d.c)
            .attr("y", (d) => d.r * h)
            .attr("width", (d) => size / data[d.r].length)
            .attr("height", h)
            .attr("fill", (d) => scale(d.v));

        if (id) {
            heatmap.svgGroup.attr("id", id);
        }
    };

    heatmap.data = function (newData) {
        if (!arguments.length) {
            return data;
        } else {
            data = newData;
            h = size / data.length;
            heatmap.setProperties();
            return heatmap;
        }
    };

    heatmap.x = function (newX) {
        if (!arguments.length) {
            return x;
        } else {
            x = newX;
            heatmap.setProperties();
            return heatmap;
        }
    };

    heatmap.y = function (newY) {
        if (!arguments.length) {
            return y;
        } else {
            y = newY;
            heatmap.setProperties();
            return heatmap;
        }
    };

    heatmap.size = function (newSize) {
        if (!arguments.length) {
            return size;
        } else {
            size = newSize;
            if (data) {
                h = size / data.length;
                heatmap.setProperties();
            }
            return heatmap;
        }
    };

    heatmap.scale = function (newScale) {
        if (!arguments.length) {
            return scale;
        } else {
            scale = newScale;
            if (data) {
                heatmap.setProperties();
            }
            return heatmap;
        }
    };

    heatmap.id = function (newId) {
        if (!arguments.length) {
            return id;
        } else {
            id = newId;
            heatmap.setProperties();
            return heatmap;
        }
    };

    return heatmap;
}

function simpleArcmap(data, m_scale, m_size, m_id, m_x, m_y) {
    const arcmap = simpleHeatmap(data, m_scale, m_size, m_id, m_x, m_y);

    function makeArc(d, size, rows, cols) {
        const angle = d3
            .scaleLinear()
            .domain([0, cols])
            .range([-Math.PI / 6, Math.PI / 6]);
        const radius = d3.scaleLinear().domain([0, rows]).range([size, 0]);

        const arcPath = d3
            .arc()
            .innerRadius(radius(d.r + 1))
            .outerRadius(radius(d.r))
            .startAngle(angle(d.c))
            .endAngle(angle(d.c + 1));

        return arcPath();
    }

    arcmap.setProperties = function () {
        var data = arcmap.data();
        var size = arcmap.size();
        var scale = arcmap.scale();
        var id = arcmap.id();
        var x = arcmap.x();
        var y = arcmap.y();

        if (!arcmap.el) {
            return;
        }

        if (!arcmap.svgGroup) {
            arcmap.svgGroup = arcmap.el.append("g");
        }

        arcmap.svgGroup.attr("transform", `translate(${x},${y})`);

        arcmap.svgGroup
            .selectAll("g")
            .data(data)
            .enter()
            .append("g")
            .selectAll("path")
            .data((d, i) =>
                d.map((val) => ({
                    r: i,
                    v: val,
                }))
            )
            .enter()
            .append("path")
            .datum(function (d, i) {
                d.c = i;
                return d;
            });

        arcmap.svgGroup
            .selectAll("g")
            .selectAll("path")
            .attr("transform", `translate(${size / 2.0},${size})`)
            .attr("d", (d) => makeArc(d, size, data.length, data[d.r].length))
            .attr("fill", (d) => scale(d.v));

        if (id) {
            arcmap.svgGroup.attr("id", id);
        }
    };

    return arcmap;
}

function arcmapLegend(
    m_scale,
    m_size,
    m_format,
    m_utitle,
    m_vtitle,
    m_x,
    m_y
) {
    let el = null;
    let utitle = m_utitle ? m_utitle : "Uncertainty";
    let vtitle = m_vtitle ? m_vtitle : "Value";
    let scale = m_scale ? m_scale : null;
    let size = m_size ? m_size : 200;
    let fmat = m_format || null;
    let x = m_x ? m_x : 0;
    let y = m_y ? m_y : 0;
    let data = null;

    const arcmap = simpleArcmap();

    var legend = function (nel) {
        el = nel;
        legend.setProperties();

        el.call(arcmap);
    };

    legend.setProperties = function () {
        if (!el) {
            return;
        }

        let tmp = data;
        if (!tmp) {
            tmp = scale.quantize().data();
        }

        const inverted = [];
        for (let i = 0; i < tmp.length; i++) {
            inverted[i] = tmp[tmp.length - i - 1];
        }

        arcmap.data(inverted);
        arcmap.scale(scale);
        arcmap.size(size);

        el.attr("class", "legend").attr("transform", `translate(${x},${y})`);

        var uncertaintyDomain =
            scale && scale.quantize ? scale.quantize().uncertaintyDomain() : [0, 1];
        const uStep =
            (uncertaintyDomain[1] - uncertaintyDomain[0]) / (inverted.length - 2);
        const uDom = d3.range(
            uncertaintyDomain[0],
            uncertaintyDomain[1] + uStep,
            uStep
        );

        const uAxisScale = d3.scalePoint().range([0, size]).domain(uDom);

        const px = size / 180;
        el.append("g")
            .attr("transform", `translate(${size + 6 * px},${28 * px})rotate(30)`)
            .call(d3.axisRight(uAxisScale).tickFormat(d3.format(fmat || "")));

        el.append("text")
            .style("text-anchor", "middle")
            .style("font-size", 13)
            .attr(
                "transform",
                "translate(" +
                (size + 30 * px) + // 20
                "," +
                (50 * px + size / 2) + // 40
                ")rotate(-60)"
            )
            .text(utitle);

        var valueDomain =
            scale && scale.quantize ? scale.quantize().valueDomain() : [0, 1];
        // const vStep = (valueDomain[1] - valueDomain[0]) / (inverted[0].length - 6);
        const vStep = (valueDomain[1] - valueDomain[0]) / 2;
        // console.log("Value domain: ", valueDomain, " V step: ", vStep)
        // const vTicks = d3.range(valueDomain[0], valueDomain[1] + vStep, vStep);
        const vTicks = d3.range(valueDomain[0], valueDomain[1]+0.0001, vStep);
        // console.log(valueDomain, vStep, inverted)
        // const vTicks = d3.range(valueDomain[0], valueDomain[1] + 5, 5);

        const vAxisScale = d3.scaleLinear().range([0, size]).domain(valueDomain);
        const valueFormat = fmat ? d3.format(fmat) : vAxisScale.tickFormat(vTicks.length - 1);
        // const valueFormat = fmat ? d3.format(fmat) : vAxisScale.tickFormat(vTicks.length - 2);

        const angle = d3.scaleLinear().domain(valueDomain).range([-30, 30]);

        const offset = 3 * px;

        const myArc = d3
            .arc()
            .innerRadius(size + offset)
            .outerRadius(size + offset + 1)
            .startAngle(-Math.PI / 6)
            .endAngle(Math.PI / 6);

        const arcAxis = el
            .append("g")
            .attr("transform", `translate(${size / 2},${size - offset})`);

        arcAxis
            .append("path")
            .attr("fill", "black")
            .attr("stroke", "transparent")
            .attr("d", myArc);

        const labelEnter = arcAxis
            .selectAll(".arc-label")
            .data(vTicks)
            .enter()
            .append("g")
            .attr("class", "arc-label")
            .attr(
                "transform",
                (d) =>
                    "rotate(" +
                    angle(d) +
                    ")translate(" +
                    0 +
                    "," +
                    (-size - offset) +
                    ")"
            );

        labelEnter
            .append("text")
            .style("font-size", "11")
            .style("text-anchor", "middle")
            .attr("y", -10)
            .text(valueFormat);

        labelEnter
            .append("line")
            .attr("x1", 0.5)
            .attr("x2", 0.5)
            .attr("y1", -6)
            .attr("y2", 0)
            .attr("stroke", "#000");

        el.append("text")
            .style("text-anchor", "middle")
            .style("font-size", 13)
            .attr("x", size / 2)
            .attr("y", -30)
            .text(vtitle);
    };

    legend.data = function (newData) {
        if (!arguments.length) {
            return data;
        } else {
            data = newData;
            legend.setProperties();
            return legend;
        }
    };

    legend.scale = function (s) {
        if (!arguments.length) {
            return scale;
        } else {
            scale = s;
            legend.setProperties();
            return legend;
        }
    };

    legend.size = function (s) {
        if (!arguments.length) {
            return size;
        } else {
            size = s;
            legend.setProperties();
            return legend;
        }
    };

    legend.format = function (f) {
        if (!arguments.length) {
            return fmat;
        } else {
            fmat = f;
            legend.setProperties();
            return legend;
        }
    };

    legend.x = function (nx) {
        if (!arguments.length) {
            return x;
        } else {
            x = nx;
            legend.setProperties();
            return legend;
        }
    };

    legend.y = function (ny) {
        if (!arguments.length) {
            return y;
        } else {
            y = ny;
            legend.setProperties();
            return legend;
        }
    };

    legend.utitle = function (t) {
        if (!arguments.length) {
            return utitle;
        } else {
            utitle = t;
            legend.setProperties();
            return legend;
        }
    };

    legend.vtitle = function (t) {
        if (!arguments.length) {
            return vtitle;
        } else {
            vtitle = t;
            legend.setProperties();
            return legend;
        }
    };

    return legend;
}

export default class LegendPlot  extends Component {
    constructor(props) {
        super();
        this.gref = React.createRef()
        this.state = {
            size: null
        }
    }

    static getDerivedStateFromProps(nextProps, prevState) {
        /*
            Whenever props change, this function will be invoked.
         */
        const {size, translate, consistencyColorScale, movingMagnitude, movingVariance, vsupxScale, vsupyScale,
        angleScale, radiusScale} = nextProps;

        return {size, translate, consistencyColorScale, movingMagnitude, movingVariance, vsupxScale, vsupyScale,
        angleScale, radiusScale};
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        if (!this.props.consistencyColorScale)
            return null

        if (prevState.consistencyColorScale !== this.props.consistencyColorScale) {
            return true
        }
        if (prevState.movingMagnitude !== this.props.movingMagnitude) {
            return true
        }
    }

    render() {
        // Remove previous g
        d3.select(this.gref.current).select('g').remove()
        let g = d3.select(this.gref.current).append('g')

        var legend = arcmapLegend(null, null, null, 'Inconsistency', 'Edit Magnitude')
            .scale(this.props.consistencyColorScale)
            .size(this.props.size)

        g.call(legend)

        // Moving metric
        // console.log(this.props.vsupxScale(this.props.movingMagnitude), this.props.vsupyScale(this.props.movingVariance))

        // const angle = Math.PI * 0.45  // (0 ~ pi * 2) 0.5 is the middle, and we are between 1 ~ 0
        // Map `movingMagnitude` to angle and `movingVariance` to radius within bounds
        const angle = this.props.angleScale(this.props.movingMagnitude);
        let radius = this.props.radiusScale(this.props.movingVariance);
        //
        // // Convert polar coordinates (angle, radius) to Cartesian coordinates (x, y)
        const xPosition = (this.props.size / 2) + radius * Math.cos(angle);
        const yPosition = this.props.size - radius * Math.sin(angle); // Adjust y for SVG coordinate system

        // console.log("Angle: ", angle, "  Radius: ", radius)
        // console.log(this.props.movingMagnitude, "xpos: ", xPosition)
        // console.log(this.props.movingVariance, "ypos: ", yPosition


        g.append('circle')
            .attr('cx', xPosition)
            .attr('cy', yPosition)
            .attr('r', 2)
            .attr('fill', 'lightcoral')

        // for (var i = 0; i < 200; i++) {
        //     for (var j = 0; j < 1000; j++) {
        //         const angle = this.props.angleScale(i * 0.1);
        //         let radius = this.props.radiusScale(j * 0.001);
        //
        //         // Convert polar coordinates (angle, radius) to Cartesian coordinates (x, y)
        //         const xPosition = (this.props.size / 2) + radius * Math.cos(angle);
        //         const yPosition = this.props.size - radius * Math.sin(angle); // Adjust y for SVG coordinate system
        //
        //         g.append('circle')
        //             .attr('cx', xPosition)
        //             .attr('cy', yPosition)
        //             .attr('r', 1)
        //     }
        // }


        return (
            <g ref={this.gref}
               transform={`translate(${25}, ${50})`}
               width={110}
               height={110}/>
        )
    }
}