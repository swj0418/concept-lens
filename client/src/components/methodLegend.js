import React, {Component} from "react";
import * as d3 from "d3"
import {forEach} from "react-bootstrap/ElementChildren";
import {splitExperimentName} from "../helper_functions/splitExperimentName";

export default class MethodLegend  extends Component {
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

    // shouldComponentUpdate(nextProps, nextState, nextContext) {
    //     if (nextProps.experimentNames) {
    //         return true
    //     }
    //
    //     return false
    // }

    drawComponent() {
        let g = d3.select(this.gref.current)
        g.selectAll('g').remove()

        // Determine present methods
        let methodNames = []
        let applicationNames = []
        let layerSplit = [0, 1, 2]
        for(const expName of this.state.experimentNames) {
            console.log(expName)
            let [domainName, methodName, applicationName, layerName, layerSubName] = splitExperimentName(expName)
            methodNames.push(methodName + ' ' + applicationName)
        }
        methodNames = [...new Set(methodNames)]
        // console.log("Displaying methods: ", methodNames)

        // Nest
        let nestedData = []
        for (const expName of methodNames) {
            for (const layer of layerSplit) {
                const item = {'expName': expName, 'layer': layer}
                nestedData.push(item)
            }
        }
        nestedData = d3.groups(nestedData, d => d.expName)

        // Scales
        let xScale = d3.scaleBand()
            .domain(methodNames)
            .range([0, 740])
            .paddingInner(0.05)
            .paddingOuter(0.05)

        let cont = g.selectAll()
            .data(nestedData)
            .enter()
            .append('g')
            .attr('width', xScale.bandwidth())
            .attr('height', 20)
            .attr('transform', (d, i) => {
                return `translate(${xScale(d[0])}, 50)`
            })

        cont.selectAll('g')
            .data(d => d[1])
            .enter()
            .append('rect')
            .attr('fill', (d, i) => {
                // return d3.hcl(
                //     this.state.methodColorScale.hue(d.expName),
                //     this.state.methodColorScale.chr(d.expName),
                //     this.state.methodColorScale.lum(d.expName) - this.state.methodColorScale.lay(i),
                // )
                const methodName = d.expName.split(" ")[0]

                return d3.hcl(
                    this.state.methodColorScale.hue(methodName),
                    this.state.methodColorScale.chr(methodName),
                    this.state.methodColorScale.lum(methodName) - this.state.methodColorScale.lay(i),
                )
            })
            .attr('width', xScale.bandwidth() / layerSplit.length)
            .attr('height', 20)
            .attr('transform', (d, i) => {
                return `translate(${(xScale.bandwidth() / 3) * i}, 0)`
            })


        cont.append('text')
            .text(d => d[0])
            .attr('text-anchor', 'start')
            .attr('line-height', 100)
            .attr('text-align', 'center')
            .attr('font-size', '15px')
            .attr('transform', `translate(${0}, ${-20 / 2 + 3})`)

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