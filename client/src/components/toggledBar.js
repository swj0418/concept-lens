import React, {Component} from "react";
import * as d3 from "d3"
import {accumulateLeafNodesBudget} from "../helper_functions/accumulateLeafNodes";
import {splitExperimentName} from "../helper_functions/splitExperimentName";


export default class ToggledBar extends Component {
    constructor(props) {
        super();
        this.gref = React.createRef()
        this.state = {
            leafNodes: []
        }
    }

    static getDerivedStateFromProps(nextProps, prevState) {
        const { positionalHierarchyDirection, methodColorScale, size, translate } = nextProps;

        let leafNodes = [];
        try {
            leafNodes = accumulateLeafNodesBudget(positionalHierarchyDirection) || [];
        } catch (e) {
            // console.error("Error accumulating leaf nodes:", e);
        }

        return { positionalHierarchyDirection, leafNodes, size, translate, methodColorScale };
    }

    componentDidMount() {
        this.drawComponent();
    }

    componentDidUpdate(prevProps, prevState) {
        if (
            prevState.positionalHierarchyDirection !== this.state.positionalHierarchyDirection ||
            prevState.leafNodes !== this.state.leafNodes
        ) {
            this.drawComponent();
        }
    }

    drawComponent() {
        if (!this.state.leafNodes)
            return

        // d3.select(this.gref.current).selectAll('rect').clear()
        const layerPatterns = {
          early: 'url(#diagonal-lines)', // Apply 빗금 pattern
          middle: 'url(#cross-hatch)',   // Apply X's pattern
          late: 'url(#solid-color)',     // Apply solid color
        };

        const { leafNodes, size } = this.state;
        const { methodColorScale } = this.props;

        d3.select(this.gref.current)
          .selectAll('g')
          .data(leafNodes)
          .join('g') // Create groups for each leaf node
          .attr('transform', (d, i) => `translate(${(size[0] / leafNodes.length) * i}, 0)`)
          .each(function (d, i) {
            // Use splitExperimentName to extract details
            const [domainName, methodName, applicationName, layerName] = splitExperimentName(d.expName);

            // console.log("TB", d.expName)
            // Compute color based on methodName and layerName
            const color = d3.hcl(
              methodColorScale.hue(methodName),
              methodColorScale.chr(methodName),
              methodColorScale.lum(methodName) - methodColorScale.lay(layerName)
              // methodColorScale.lum(methodName) - 10
            );

            // Determine pattern based on layerName
            const pattern = layerName === 'early' ? 'url(#horizontal-lines)' :
                            layerName === 'middle' ? 'url(#vertical-lines)' : null;

            // Append background color rectangle
            d3.select(this).append('rect')
              .attr('width', (size[0] / leafNodes.length) / 1.0)
              .attr('height', size[1])
              .attr('fill', color)

          });
    }

    render() {
        return (
            <g ref={this.gref} transform={`translate(${this.state.translate[0]}, ${this.state.translate[1]})`} onClick={this.props.onclick} />
        );
    }
}