import React, {Component} from "react";
import {hierarchyScale} from "../scales/hierarchyScale";
import * as d3 from "d3"
import {divergingDistributionColorScale} from "../scales/divergingDistributionColorScale";
import {accumulateLeafNodes} from "../helper_functions/accumulateLeafNodes";

export default class VerticalIciclePlot extends Component {
    constructor(props) {
        super();
        this.gref = React.createRef()
        this.state = {
            parentG: null,
            positionalHierarchyData: null,
            size: null,
            paddingInner: 0.01,
            paddingOuter: 0.01,
            experimentNames: null,
            timestamp: 0,
            brush: null
        }

        var filter = d3.select(this.gref.current).append("defs")
            .append("filter")
            .attr("id", "shadowr")
            .attr("x", "-5%")
            .attr("y", "-5%")
            .attr("width", "250%")
            .attr("height", "250%");

        filter.append("feOffset")
            .attr("result", "offOut")
            .attr("in", "SourceAlpha")
            .attr("dx", "10")
            .attr("dy", "0");

        filter.append("feGaussianBlur")
            .attr("result", "blurOut")
            .attr("in", "offOut")
            .attr("stdDeviation", "5");

        filter.append("feBlend")
            .attr("in", "SourceGraphic")
            .attr("in2", "blurOut")
            .attr("mode", "normal");

        // Action Binding
        this.brushAction = this.brushAction.bind(this)
        this.brushActionDuring = this.brushActionDuring.bind(this)
        this.clickAction = this.clickAction.bind(this)
        this.clickBarAction = this.clickBarAction.bind(this)
    }

    static getDerivedStateFromProps(nextProps, prevState) {
        /*
            Whenever props change, this function will be invoked.
         */
        const {parentG, positionalHierarchyData, colorScale,
            // avgMagnitude, avgStd, m_std_interval, u_std_interval,
            size, translate, visDepth, experimentNames} = nextProps;

        if (!positionalHierarchyData)
            return null

        // if (!hierarchyData)
        //     return null

        let depthScale = d3.scaleBand()
            .domain(d3.range(visDepth + 1))
            .range([0, size[0]])

        return {parentG, positionalHierarchyData, depthScale, colorScale, size, translate, visDepth, experimentNames};
    }

    drawPlot() {
        console.log("Drawing vertical plot")
        const visDepth = this.props.visDepth

        let nodeG = d3.select(this.gref.current).selectAll("g") // Does not draw twice anymore. But why?
            .data([this.state.positionalHierarchyData], d => d.name) // Supply a key to match the data.
            .join('g')
            .attr('transform', d => `translate(0, ${d.position})`)
            .classed('iciclenode', true)

        let rootNodeG = d3.select(this.gref.current).selectAll("g")

        for(let i = 0; i < visDepth; i++) {
            nodeG = nodeG.selectAll()
                .data(d => d.children, d => d.name)
                .join('g')
                .attr('transform', d => `translate(0,${d.position})`)
                .classed('iciclenode', true)
        }

        // Give IDs
        d3.select(this.gref.current).selectAll('.iciclenode')
            .classed('verticalnode', true)

        const S = 0.15
        d3.select(this.gref.current).selectAll('.iciclenode')
            .append('rect')
            .attr('x', d => {
                // return this.state.depthScale(d.depth)

                    const Sx = (1 - S) / (this.props.visDepth - 1)
                    const totDepth = this.state.depthScale.bandwidth() * this.props.visDepth // Total visDepth
                    const shift = (totDepth * Sx) * d.depth
                    return shift

            })
            .attr('y', d => 0)
            .attr('width', d => {
                // return this.state.depthScale.bandwidth()

                    // Total visDepth
                    const totDepth = this.state.depthScale.bandwidth() * this.props.visDepth
                    if (d.depth === this.props.visDepth) {
                        return totDepth * S
                    }
                    return totDepth * ((1 - S) / (this.props.visDepth - 1))
            })
            .attr('height', d => d.size)
            .attr('stroke', 'black')
            .attr('stroke-width', .75)
            .attr('fill', d => {
                return this.state.colorScale(d.magnitude, d.var)
            }) // For now.
            .attr('rx', 5)
            .on('click', this.clickAction)
            .lower()

        // Visual retention
        this.nodeHighlighting(this.props.getSelectionCode().map(d => d.name))
    }

    removeItemOnce(arr, value) {
      var index = arr.indexOf(value);
      if (index > -1) {
        arr.splice(index, 1);
      }
      return arr;
    }

    nodeHighlighting(enabledSelection) {
        d3.select(this.gref.current).selectAll('.iciclenode').select('rect')
            .attr('stroke', 'black')
            .style('filter', null)
            .attr('stroke', d3.hcl(0, 0, 30))
            .attr('stroke-width', 1)

        d3.select(this.gref.current).selectAll('.iciclenode').select('rect')
            .filter(d => enabledSelection.includes(d.name))
            .filter(d => d.depth === this.state.visDepth)
            .style('filter', 'url(#shadowr)')
            // .attr('stroke', d3.hcl(120, 90, 40))
            .attr('stroke', d3.hcl(50, 100, 75))
            .attr('stroke-width', 5)

            .each(function(d) {
            const node = d3.select(this);


            // Apply styles
            node.select("rect")
                .style("filter", "url(#shadowr)")
                .attr('stroke', d3.hcl(50, 100, 75))
                .attr("stroke-width", 5);

            // node.raise();
            // node.attr("transform", d => `${node.attr("transform")} translate(-1,0)`);

            // Remove and re-add the node to raise it
            const parent = d3.select(this.parentNode);
            parent.raise();
            const pparent = d3.select(this.parentNode.parentNode);
            pparent.raise();

            node.remove(); // Remove the node
            parent.node().appendChild(this); // Re-add the node at the end
        });
    }

    getUpdatedSelectionCodes(selection) {
        let prevSelectionCode = this.props.getSelectionCode()
        let prevSelectionCodeNames = prevSelectionCode.map(d => d.name)

        if (prevSelectionCodeNames.includes(selection.name))
            prevSelectionCode.splice(prevSelectionCodeNames.indexOf(selection.name), 1)
        else
            prevSelectionCode.push(selection)
        return prevSelectionCode
    }

    clickAction(e, d) {
        // Update data and Visually update - For this I need to know which nodes are selected.
        // This information lives in the parent component.
        // I need a function to retrieve that information from parent component.
        let prevSelectionCode = this.getUpdatedSelectionCodes(d)
        let prevSelectionCodeNames = prevSelectionCode.map(d => d.name)
        this.nodeHighlighting(prevSelectionCodeNames)

        this.props.clickListener(prevSelectionCode)  // Pass it to listener for data processing
    }

    clickBarAction(e, d) {
        this.props.clickBarListener(d.name)
    }

    brushAction(e, d) {
        let brushSelection = e.selection
        let brushedItems = []

        if (!brushSelection)  // Prevents error when brush is empty.
            return

        // Get brushed Items
        d3.select(this.gref.current).selectAll('.iciclenode')
            .filter(d => d.depth === this.state.visDepth)
            .filter(function(d) {
                let start = d.absolute_position
                let end = start+d.size
                if(brushSelection[1] < start || end < brushSelection[0])
                    return false
                brushedItems.push(d)
                return true
            })

        this.nodeHighlighting(brushedItems.map(d => d.name))
        this.props.brushListener(brushedItems, brushSelection[0], brushSelection[1])
    }

    brushActionDuring(e, d) {
        let brushSelection = e.selection
        let brushedItems = []

        if (!brushSelection)  // Prevents error when brush is empty.
            return

        // Get brushed Items
        d3.select(this.gref.current).selectAll('.iciclenode')
            .filter(d => d.depth === this.state.visDepth)
            .filter(function(d) {
                let start = d.absolute_position
                let end = start+d.size
                if(brushSelection[1] < start || end < brushSelection[0])
                    return false
                brushedItems.push(d)
                return true
            })

        this.nodeHighlighting(brushedItems.map(d => d.name))
        this.props.brushListenerDuring(brushedItems, brushSelection[0], brushSelection[1])
    }

    drawBrush() {
        // Brush is drawn once during initialization, but never unattached afterward. Control this in componentDidUpd.
        let brush = d3.brushY().extent([[0,0],[this.state.depthScale.bandwidth(), this.props.size[1]]])
        brush.on('brush', this.brushActionDuring)
        brush.on('end', this.brushAction)

        d3.select(this.state.parentG).append('g')
            .attr('transform', `translate(${this.state.depthScale(this.state.visDepth) + this.state.translate[0]},
                                          ${this.state.translate[1]})`)
            .classed('brush', true)
            .classed('vertical_brush', true)
            .call(brush)
            .raise()
    }

    deleteBrush() {
        d3.select(this.state.parentG).selectAll('.vertical_brush').remove()
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        // console.log("update vertical icicle plot")

        if (prevProps.positionalHierarchyData == null && this.props.positionalHierarchyData == null) {
            return;
        }

        if (prevProps.positionalHierarchyData == null && this.props.positionalHierarchyData) {
            this.deleteBrush()
            this.drawPlot()
            this.drawBrush()
            return;
        }

        if (prevProps.positionalHierarchyData.timestamp !== this.props.positionalHierarchyData.timestamp) {
            this.deleteBrush()
            this.drawPlot()
            this.drawBrush()
            return;
        }

        // Update plot if timestamp is different.
        // if (this.state.timestamp === newTimeStamp) {
        //     // Do nothing
        // } else {
        //     this.deleteBrush()
        //     this.drawPlot()
        //     this.drawBrush()
        //     return
        // }

        if (prevState.visDepth !== this.state.visDepth) {
            this.deleteBrush()
            this.drawPlot()
            this.drawBrush()
            return
        }

        if (prevState.experimentNames !== this.state.experimentNames) {
            this.deleteBrush()
            this.drawPlot()
            this.drawBrush()
            return
        }

        if (prevState.colorScale !== this.state.colorScale) {
            this.drawPlot()
        }
    }

    render() {
        if (!this.state.translate)
            return

        return (
            <g ref={this.gref}
               transform={`translate(${this.state.translate[0]}, ${this.state.translate[1]})`}
               width={this.props.size[0]}
               height={this.props.size[1]}/>
        )
    }
}