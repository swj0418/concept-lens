import React, {Component} from "react";
import {biTreeColorScale} from "../scales/biTreeColorScale"
import * as d3 from "d3"
import {accumulateLeafNodesBudget, accumulateVisLeafNodes} from "../helper_functions/accumulateLeafNodes";


function evenlySampleArray(arr, n) {
    return Array.from({length: n}, (_, i) => arr[Math.floor(i * arr.length / n)]);
}

export default class OriginalImagePanel  extends Component {
    constructor(props) {
        super();
        this.gref = React.createRef()
        this.state = {
            parentG: null,
            size: null,
            paddingInner: 0.01,
            paddingOuter: 0.01,
            width: 130,
            methodColorScale: null
        }

        this.joinImagePanel = this.joinImagePanel.bind(this)
        this.setupCodeImages = this.setupCodeImages.bind(this)
    }

    static getDerivedStateFromProps(nextProps, prevState) {
        /*
            Whenever props change, this function will be invoked.
         */
        const {parentG, positionalHierarchyCode, size, translate, visDepth} = nextProps;

        return {parentG, positionalHierarchyCode, size, translate, visDepth};
    }

    joinImagePanel = () => {
        let biColorScale = biTreeColorScale([0,10], [40,97],4)

        let setup_single_rects = (selection, drawLine) => {
            selection
                .attr('width', this.state.width)
                .attr('height', d => d.size)
                .attr('fill', 'none')
                .attr('rx', 5)

            if (!drawLine) {
                selection.attr('stroke', 'none')
            } else {
                selection.attr('stroke', d3.hcl(0,0,30))
                .attr('fill', d => {
                    let max_d = Math.max(d.depth)
                    return 'none'
                })
            }

        }

        let enter_ops = (enter, drawLine) => {
            let g = enter.append('g')
                .attr('transform', d => `translate(0, ${d.position})`)
                .classed('rnode', true)

            g.append('rect')
                .call(setup_single_rects, drawLine)

            g.append('g')
                .filter(d => d.depth === this.state.visDepth)
                .call(this.setupCodeImages)
        }

        let update_ops = (update, drawLine) => {
            update
                .attr('transform', d => `translate(${0}, ${d.position})`)
                .select('rect') // This propagates data attached to group element to a child element, which is 'rect' element.
                .call(setup_single_rects, drawLine)
        }

        d3.select(this.gref.current).selectAll('.rnode')
            .data([this.state.positionalHierarchyCode], d => 'r-' + d.name)
            .join(
                enter => enter_ops(enter, true),
                update => update_ops(update, true),
                exit => exit.remove()
            )

        for(let i = 0; i < this.state.visDepth; i++) {
            let next_depth_g = d3.select(this.gref.current).selectAll('.rnode').filter(d => d.depth === i)
            next_depth_g.selectAll('.rnode')
                .data(d => d.children, d => 'r-' + d.name)
                .join(
                    enter => enter_ops(enter, false),
                    update => update_ops(update, false),
                    exit => exit.remove()
                )
        }
    }

    setupCodeImages(selection) {
        let methodColorScale = d3.scaleSequential(d3.schemePastel1).domain([0, 1])

        const allPositions = selection.data().map(d => ({
            rect: d,
            x: d.absolute_position,
            y: d.absolute_position,
        }));

        const minY = d3.min(allPositions, d => d.y);
        const minX = d3.min(allPositions, d => d.x)
        const topRowRects = allPositions.filter(d => d.y === minY);
        const leftColRects = allPositions.filter(d => d.x === minX);

        let glyphCodeDrawn = 0
        let glyphDirectionDrawn = 0

        let getImageLink = (codeIdx, flatIdx, treeID) => {
            let bucketPath = `http://localhost:${this.props.port}/served_data/${this.props.experimentNames[0]}/`
            return bucketPath + 'codes/' + `${codeIdx}.jpg`
        }

        let insertImage = (selection, imageLink, imageSize, xPos, yPos) => {
            // Sets up single image
            selection
                .attr('onerror', "this.style.display='none'")
                .attr("xlink:href", imageLink)
                .attr('transform', `translate(${xPos}, ${yPos})`)
                .attr('width', d => imageSize)
                .attr('height', d => imageSize)
                .attr('opacity', 0)
                .on('click', function (event, d) {
                    d3.select(this).raise().transition().duration(500)
                        .attr('width', d3.select(this).attr('width') == imageSize ? imageSize * 4 : imageSize)
                        .attr('height', d3.select(this).attr('height') == imageSize ? imageSize * 4 : imageSize)
                })
                .transition().duration(10) // Adjust the duration as needed
                .attr('opacity', 1) // Final opacity
        }

        let topG = selection.filter(d => (d.depth === this.state.visDepth)) // Draw only for the top boxes.
        topG
            .attr('width', this.state.width)
            .attr('height', d => d.size)

        // Number of topGs
        let topGcount = selection.size()
        let methodDrawnCount = -1

        // scales for individual boxes
        topG.each(function(d, i) {
            let topRow = false
            let leftCol = false
            for (let k in topRowRects) {
                if (d === topRowRects[k].rect) {
                    topRow = true
                }
            }

            for (let k in leftColRects) {
                if (d === leftColRects[k].rect) {
                    leftCol = true
                }
            }

            const width = 125
            const height = d.size
            // const height = d.size
            // const width = d.size, height = d.size
            const selection = d3.select(this)
            let nodeData = selection.data()[0]

            let imageSize = 120

            // Determine how many images can fit into each box.
            let horizontalCount = 1
            // let verticalCount = Math.floor(height / imageSize)  // Code budget

            // Determine how many images can fit into each box.
            let verticalCount = Math.floor(height / imageSize)  // Code budget
            if (topRow) {
                verticalCount = Math.floor((height - 24) / imageSize)  // Code budget
            }
            console.log("Ori Tree vertical, horizontal budget", verticalCount, horizontalCount)

            // Use width and height to determine padding.
            const horizontalPadding = (width - (imageSize * horizontalCount)) / 2
            const verticalPadding = (height - (imageSize * verticalCount)) / 2

            const horizontalScale = d3.scaleBand().domain(Array.from({length: horizontalCount}, (_, i) => i))
                .range([horizontalPadding, width - horizontalPadding]).paddingOuter(0.1).paddingInner(0.1)

            const verticalScale = d3.scaleBand().domain(Array.from({length: verticalCount}, (_, i) => i))
                .range([verticalPadding, height - verticalPadding]).paddingOuter(0.1).paddingInner(0.1)

            // Leaf node indices
            // let codeLeaf = accumulateVisLeafNodes(nodeData)
            let codeLeaf = accumulateLeafNodesBudget(nodeData)

            if (codeLeaf.length < verticalCount) {
                verticalCount = codeLeaf.length
            }

            let codeSample = evenlySampleArray(codeLeaf, verticalCount)

            for(var v = 0; v < verticalCount; v++) {
                const imageLink = getImageLink(codeSample[v].name, 0, 0)
                let xPos = horizontalScale(0)
                let yPos = verticalScale(v)
                if (topRow) yPos += 20
                if (codeSample[v]) {
                    selection
                        .append('image')
                        .call(insertImage, imageLink,
                            verticalScale.bandwidth(),
                            xPos,
                            yPos)
                }
            }
        })
    }

    componentDidUpdate(prevProps, prevState, snapshot) {
        if (prevState.positionalHierarchyCode !== this.state.positionalHierarchyCode) {
            this.joinImagePanel()
        }
    }

    render() {
        if(!this.state.translate)
            return

        return (
            <g ref={this.gref}
               transform={`translate(${this.state.translate[0]}, ${this.state.translate[1]})`}
               width={this.state.size[0]}
               height={this.state.size[1]}/>
        )
    }
}