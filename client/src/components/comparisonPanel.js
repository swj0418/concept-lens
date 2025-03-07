import React, {Component} from "react";
import * as d3 from "d3"
import {forEach} from "react-bootstrap/ElementChildren";
import {splitExperimentName} from "../helper_functions/splitExperimentName";

export default class MethodComparison  extends Component {
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
        const {experimentNames, methodColorScale, size, translate} = nextProps;

        return {experimentNames, methodColorScale, size, translate};
    }

    drawComponent() {
        let svg = d3.select(this.gref.current)
        svg.selectAll('g').remove()
        svg.selectAll('line').remove()

        // Draw mean, min, max for all experiment of the same DOMAIN, LAYER.



        // Axis
        d3.select(this.gref.current).append('line')
            .attr('x1', 0)
            .attr('y1', 0)
            .attr('x2', 0)
            .attr('y2', 100)
            .style('stroke', 'black')
            .style('stroke-width', 2)
            .style('opacity', 0.3)

        d3.select(this.gref.current).append('line')
            .attr('x1', 0)
            .attr('y1', 100)
            .attr('x2', 100)
            .attr('y2', 100)
            .style('stroke', 'black')
            .style('stroke-width', 2)
            .style('opacity', 0.3)

    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        /*
        Invoke render function when
            1. Current state has experimentNames

        Invoke re-render function when
            1. Previous state exists and current state is different from previous state

        Otherwise, do not render
         */

        if (this.state.experimentNames == null)
            return false

        this.drawComponent()
    }

    render() {
        return (
            <svg ref={this.gref} width={this.state.size} height={100}/>
        )
    }
}