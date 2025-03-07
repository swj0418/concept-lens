import './App.css';
import {Component, createRef} from "react";
import VerticalIciclePlot from "./components/verticalIliclePlot";
import HorizontalIciclePlot from "./components/horizontalIciclePlot";
import OriginalImagePanel from "./components/originalImagePanel";
import ToggledBar from "./components/toggledBar"

import {cullHierarchyByNameTruncatedTree, cullHierarchyByNameAndQuantity} from "./helper_functions/cullHierarchy";
import {accumulateLeafNodes, accumulateLeafNodesBudget} from "./helper_functions/accumulateLeafNodes";
import {accumulateQuantity} from "./helper_functions/accumulateQunantity";
import {hierarchyScale} from "./scales/hierarchyScale";
import BiTree from "./components/biTree";
import SettingView from "./pages/settingView/settingView";

import divergingDistributionColorLegend from "./scales/divergingDistributionColorLegend"
import * as d3 from "d3";
// import * as d3c from "d3-color"
import LegendPlot from "./components/legendPlot";
import MethodLegend from "./components/methodLegend";
import MethodComparison from "./components/comparisonPanel";
import MethodLegendV2 from "./pages/methodLegend/methodLegendV2"
import {Button} from "react-bootstrap";
import {Divider, Slider} from "antd";
import IntervalController from "./components/intervalController";

import * as vsup from "vsup/src"

// Resources
import MethodColorLegend from "./ConceptLensMethodColorLegendV2.svg"

import colors from "d3-color";
import {gatherWeaveScores} from "./helper_functions/gatherWeaveScores";

function colorsj(specifier) {
    var n = specifier.length / 6 | 0, colors = new Array(n), i = 0;
    while (i < n) colors[i] = "#" + specifier.slice(i * 6, ++i * 6);
    return colors;
}

function ramp(range) {
    var n = range.length;
    return function(t) {
        t = Math.abs(1 - t)
        return range[Math.max(0, Math.min(n - 1, Math.floor(t * n)))];
    };
}

let viridis = ramp(colorsj("44015444025645045745055946075a46085c460a5d460b5e470d60470e6147106347116447136548146748166848176948186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f854240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f881fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad8128ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc743fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc8635ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21ad8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725"));

function interpolateCividisInverse(x) {
    x = Math.abs((1 - x))
    x = Math.max(0, Math.min(1, x));
    return `rgb(${[
        -4.54 - x * (35.34 - x * (2381.73 - x * (6402.7 - x * (7024.72 - x * 2710.57)))),
        32.49 + x * (170.73 + x * (52.82 - x * (131.46 - x * (176.58 - x * 67.37)))),
        81.24 + x * (442.36 - x * (2482.43 - x * (6167.24 - x * (6614.94 - x * 2475.67))))
    ].map(Math.floor).join(", ")})`;
}


export default class App extends Component {
    constructor() {
        super();
        this.svgRef = createRef()
        this.state = {
            experimentNames: [],
            methodBlindMode: 'false',
            featureProcessingMethod: 'end',
            clusteringMethod: 'complete',
            pairwiseMetric: 'cosine',

            // Original, not truncated, unchanging data
            codeHierarchyInitialData: null,
            directionHierarchyInitialData: null,

            // Original, truncated, unchanging data
            codeHierarchyBitreeData: null,
            directionHierarchyBitreeData: null,

            // BiTree
            positionalHierarchyDirectionBitree: null,
            positionalHierarchyCodeBitree: null,

            // For Consistency metric in icicle plot
            positionalHierarchyDirectionIcicle: null,
            positionalHierarchyCodeIcicle: null,

            // Vis
            height: null,
            width: null,
            icicleSize: null,
            settingWidth: 0,
            toggledBarHeight: 12,
            originalImagePlotSize: null,
            oriGap: 15,
            imageSize: null,
            visDepth: null,
            truncatedTree: true,

            // Selected Direction & Code items
            selectedDirectionNodes: [],
            selectedDirectionLeaves: [],
            selectedCodeNodes: [],
            selectedCodeLeaves: [],

            // Hyper-parameters for the visualization.
            m_std_interval: .4,
            u_std_interval: .4,

            // Variably
            weaveMax: 10,

            // State
            justReclustered: false,

            // Moving state
            movingMagnitude: 0,
            movingVariance: 0,

            // Contributions
            contributions: [],

            methodColorScale: {
                'hue': d3.scaleOrdinal(
                    ['vac', 'sefakmc', 'ganspacekmc', 'ae', 'svmw',
                        'ganspacekmc_male', 'ganspacekmc_female'
                    ],
                    [15, 210, 80, 50, 280,
                        80, 80
                    ] // Slight adjustments to hues for better contrast
                ),
                'chr': d3.scaleOrdinal(
                    ['vac', 'sefakmc', 'ganspacekmc', 'ae', 'svmw',
                        'ganspacekmc_male', 'ganspacekmc_female'
                    ],
                    [70, 75, 80, 65, 85,
                        80, 80
                    ] // Increased chroma for Cobalt Blue and Warm Red
                ),
                'lum': d3.scaleOrdinal(
                    ['vac', 'sefakmc', 'ganspacekmc', 'ae', 'svmw',
                        'ganspacekmc_male', 'ganspacekmc_female'
                    ],
                    [45, 55, 65, 50, 60,
                        20, 80
                    ] // Larger luminance contrast for middle layers
                ),
                'lay': d3.scaleOrdinal(
                    ['early', 'middle', 'late'],
                    [0, 15, 30] // Middle layers now have higher contrast
                ),
            },

            // Infra
            port: 37203
        }

        // Vertical
        this.verticalIcicleListener = this.verticalIcicleListener.bind(this)
        this.verticalIcicleListenerClick = this.verticalIcicleListenerClick.bind(this)
        this.verticalIcicleListenerBarClick = this.verticalIcicleListenerBarClick.bind(this)
        this.getSelectionCode = this.getSelectionCode.bind(this)
        this.clearSelectionCode = this.clearSelectionCode.bind(this)

        // Horizontal
        this.horizontalIcicleListener = this.horizontalIcicleListener.bind(this)
        this.horizontalIcicleListenerClick = this.horizontalIcicleListenerClick.bind(this)
        this.horizontalIcicleListenerBarClick = this.horizontalIcicleListenerBarClick.bind(this)
        this.getSelectionDirection = this.getSelectionDirection.bind(this)
        this.clearSelectionDirection = this.clearSelectionDirection.bind(this)

        // Intermediate listener
        this.verticalIcicleListenerDuring = this.verticalIcicleListenerDuring.bind(this)
        this.horizontalIcicleListenerDuring = this.horizontalIcicleListenerDuring.bind(this)

        // Button listeners
        this.toggledBarClickListener = this.toggledBarClickListener.bind(this)
        this.reclusterButtonClickListener = this.reclusterButtonClickListener.bind(this)
        this.resetButtonClickListener = this.resetButtonClickListener.bind(this)
        this.mSliderListener = this.mSliderListener.bind(this)
        this.uSliderListener = this.uSliderListener.bind(this)
        this.imageSizeSliderListener = this.imageSizeSliderListener.bind(this)
    }

    static getDerivedStateFromProps(nextProps, prevState) {
        /*
            Whenever props change, this function will be invoked.
         */
        const {experimentNames, methodBlindMode, featureProcessingMethod,
            clusteringMethod, pairwiseMetric, truncatedTree,
            visDepth, height, width, icicleSize, originalImagePlotSize, imageSize, toggledBarHeight} = nextProps;

        return {experimentNames, methodBlindMode, featureProcessingMethod,
            clusteringMethod, pairwiseMetric, truncatedTree,
            visDepth, height, width, icicleSize, originalImagePlotSize, imageSize, toggledBarHeight};
    }

    getSelectionCode() {
        return this.state.selectedCodeNodes
    }

    getSelectionCodeLeaves() {
        return this.state.selectedCodeLeaves
    }

    getSelectionDirection() {
        return this.state.selectedDirectionNodes
    }

    getSelectionDirectionLeaves() {
        return this.state.selectionDirectionLeaves
    }

    clearSelectionCode() {
        this.setState({selectedCodeNodes: []})
    }

    clearSelectionCodeLeaves() {
        this.setState({selectedCodeLeaves: []})
    }

    clearSelectionDirection() {
        this.setState({selectedDirectionNodes: []})
    }

    clearSelectedDirectionLeaves() {
        this.setState({selectedCodeLeaves: []})
    }

    // Older method
    gatherUnderNodeLeaves(selectedNodes, printMode=false) {
        // filter hierarchy data - all child nodes
        let child_nodes = []
        for (var x in selectedNodes) {
            let tmp_nodes = accumulateLeafNodesBudget(selectedNodes[x]).map(d => d.name)
            child_nodes.push(...tmp_nodes)
            if (printMode)
                console.log(tmp_nodes)
        }

        return child_nodes
    }

    gatherUnderBrushLeaves(selectedNodes, start, end, printMode=false) {
        // Determine range to retain
        let totalSize = 0
        for(var x in selectedNodes) {
            totalSize += selectedNodes[x].size
        }

        // absolute start & end
        let absStart = selectedNodes[0].absolute_position
        let absEnd = selectedNodes[selectedNodes.length - 1].absolute_position +
            selectedNodes[selectedNodes.length - 1].size

        // Proportions
        let startSkip = (start - absStart) / totalSize
        let endSkip = (absEnd - end) / totalSize

        // filter hierarchy data - all child nodes
        let child_nodes = []
        let child_nodes_sep = [] // child nodes separated.
        for (var y in selectedNodes) {
            let tmp_nodes = accumulateLeafNodesBudget(selectedNodes[y]).map(d => d.name)
            child_nodes.push(...tmp_nodes)
            child_nodes_sep.push(tmp_nodes)
            if (printMode)
                console.log(tmp_nodes)
        }

        // Skip start and end - These numbers are correct.
        let totalLeafNodes = child_nodes.length
        let startSkipNodeLength = Math.floor(startSkip * totalLeafNodes)
        let endSkipNodeLength = Math.ceil((1 - endSkip) * totalLeafNodes)
        // console.log(child_nodes)
        // console.log(totalLeafNodes, startSkipNodeLength, endSkipNodeLength)

        if (startSkipNodeLength === -1)
            startSkipNodeLength = 0

        // console.log(child_nodes, startSkipNodeLength, endSkipNodeLength, child_nodes.slice(startSkipNodeLength, endSkipNodeLength))

        return child_nodes.slice(startSkipNodeLength, endSkipNodeLength)
    }

    computeFilteredHierarchy(hierarchy, childNodes, orientation) {
        /*
            Computes filtered hierarchy that is used by biTree. Always use initialized data. The data is cloned because
            the culling is in-place operation.
         */

        // a filtered hierarchy for BiTree
        let clonedHierarchy = structuredClone(hierarchy)
        let filteredHierarchy = null
        if (orientation === 'vertical') {
            filteredHierarchy = cullHierarchyByNameTruncatedTree(clonedHierarchy, childNodes)[1]
        } else if (orientation === 'horizontal') {
            filteredHierarchy = cullHierarchyByNameTruncatedTree(clonedHierarchy, childNodes)[1]
        } else {
            console.error("There is no such orientation.")
        }

        // Re-accumulate quantity of the filtered tree.
        accumulateQuantity(filteredHierarchy)

        return filteredHierarchy
    }

    computeCulledHierarchy(hierarchy, childNodes, subsets) {
        let clonedHierarchy = structuredClone(hierarchy)
        let culledHierarchy = cullHierarchyByNameAndQuantity(clonedHierarchy, childNodes, subsets)[1]

        accumulateQuantity(culledHierarchy)
        return culledHierarchy
    }

    computePositionalHierarchy(hierarchy, orientation) {
        /*
            Positional hierarchy for biTree.
         */
        let c_hierarchy = structuredClone(hierarchy)
        let positionalHierarchy = null
        if (orientation === 'vertical') {
            this.state.verticalHierarchyScale.domain(c_hierarchy)
            positionalHierarchy = this.state.verticalHierarchyScale(c_hierarchy, true, this.state.visDepth)
        } else if (orientation === 'horizontal') {
            this.state.horizontalHierarchyScale.domain(c_hierarchy)
            positionalHierarchy = this.state.horizontalHierarchyScale(c_hierarchy, true, this.state.visDepth)
        } else {
            console.error("There is no such orientation.")
        }

        return positionalHierarchy
    }

    async verticalIcicleUpdateLogic(newChildNodes) {
        newChildNodes = newChildNodes.filter(function( element ) { return element !== undefined })
        console.log("Vertical icicle listener: ", newChildNodes)

        // Request coherence data for sub-selection -- update the other tree.
        // let directionData = null
        let tree = null
        let directionData = null
        if (this.state.justReclustered) {
            console.log("Selecting code after re-clustering")
            directionData = await this.fetchDirectionHierarchyInitialization([])
            tree = directionData.tree
            directionData.contributions = []
            directionData.magmin = 0
            directionData.magmax = 1
            // Contribution need to be input

        } else {
            directionData = await this.fetchCodeHierarchySelectionData(
                this.state.directionHierarchyInitialData,
                newChildNodes,
                this.state.selectedDirectionLeaves
            )
            tree = directionData.directionTree
        }

        // Re-compute hierarchy for the other tree for the icicle plot.
        let tmpHScale = hierarchyScale()
            .domain(tree)
            .range([0, this.state.width - this.state.icicleSize - this.state.settingWidth])
            .paddingInner(0.000)
            .paddingOuter(0.000)
        let positionalHierarchyDirectionIcicle = tmpHScale(tree, true, this.state.visDepth)

        // Data structure for biTree
        let codeTree = structuredClone(this.state.codeHierarchyBitreeData)
        let filteredHierarchy = this.computeFilteredHierarchy(codeTree, newChildNodes, 'vertical')
        let positionalHierarchyCodeBitree = this.computePositionalHierarchy(filteredHierarchy, 'vertical')

        let avgMagnitude = directionData.avgMagnitude
        let avgStd = directionData.avgStd
        let magsStd = directionData.magsStd
        let stdsStd = directionData.stdsStd
        let contributions = directionData.contributions
        let magmin = directionData.magmin
        let magmax = directionData.magmax

        return {positionalHierarchyCodeBitree, positionalHierarchyDirectionIcicle, avgMagnitude, avgStd, magsStd, stdsStd,
        contributions, magmin, magmax}
    }

    async verticalIcicleListener(selectedCodeNodes, start, end) { // console.log("Action from vertical icicle plot")
        /*
            Selection given in the parameter are truncated leaf nodes.
         */
        // Reset previous selection
        this.setState({selectedCodeNodes: []})
        this.setState({selectedCodeLeaves: []})

        // Gather child nodes
        let newChildNodes = this.gatherUnderBrushLeaves(selectedCodeNodes, start, end)

        // Set new selection
        this.setState({selectedCodeNodes: selectedCodeNodes})
        this.setState({selectedCodeLeaves: newChildNodes})

        let updated = await this.verticalIcicleUpdateLogic(newChildNodes)

        if (this.state.justReclustered) {
            this.setupVisualization()
        }

        this.setState({
            selectedCodeNodes: selectedCodeNodes, // Visual nodes / intermediate nodes
            selectedCodeLeaves: newChildNodes, // Actual leaf nodes
            positionalHierarchyCodeBitree: updated.positionalHierarchyCodeBitree, // For BiTree
            positionalHierarchyDirectionIcicle: updated.positionalHierarchyDirectionIcicle, // For Icicle plot
            justReclustered: false,
            contributions: updated.contributions,
            magmin: updated.magmin,
            magmax: updated.magmax
        })
    }

    async verticalIcicleListenerDuring(selectedCodeNodes, start, end) { // console.log("Action from vertical icicle plot")
        /*
            Selection given in the parameter are truncated leaf nodes.
         */
        // Reset previous selection
        this.setState({selectedCodeNodes: []})
        this.setState({selectedCodeLeaves: []})

        // Gather child nodes
        let newChildNodes = this.gatherUnderBrushLeaves(selectedCodeNodes, start, end)

        // Set new selection
        this.setState({selectedCodeNodes: selectedCodeNodes})
        this.setState({selectedCodeLeaves: newChildNodes})

        newChildNodes = newChildNodes.filter(function( element ) { return element !== undefined })
        console.log("Vertical icicle listener: ", newChildNodes)

        // Data structure for biTree
        let codeTree = structuredClone(this.state.codeHierarchyBitreeData)
        let filteredHierarchy = this.computeFilteredHierarchy(codeTree, newChildNodes, 'vertical')
        let positionalHierarchyCodeBitree = this.computePositionalHierarchy(filteredHierarchy, 'vertical')

        this.setState({
            selectedCodeNodes: selectedCodeNodes, // Visual nodes / intermediate nodes
            selectedCodeLeaves: newChildNodes, // Actual leaf nodes
            positionalHierarchyCodeBitree: positionalHierarchyCodeBitree, // For BiTree
            justReclustered: false
        })
    }

    async verticalIcicleListenerClick(selectedCodeNodes) {
        /*
            Selection given in the parameter are truncated leaf nodes.

            clickedNodes: name of nodes (intermediate or terminal), and need to be accumulated afterward. The reason
            for doing this instead of passing around leaf node names is to make it easier to control visual highlights.
         */

        // Reset previous selection
        this.setState({selectedCodeNodes: []})
        this.setState({selectedCodeLeaves: []})

        // Gather child nodes
        let newChildNodes = this.gatherUnderNodeLeaves(selectedCodeNodes)

        // Set new selection
        this.setState({selectedCodeNodes: selectedCodeNodes})
        this.setState({selectedCodeLeaves: newChildNodes})

        let updated = await this.verticalIcicleUpdateLogic(newChildNodes)

        if (this.state.justReclustered) {
            this.setupVisualization()
        }

        this.setState({
            selectedCodeNodes: selectedCodeNodes, // Visual nodes / intermediate nodes
            selectedCodeLeaves: newChildNodes, // Actual leaf nodes
            positionalHierarchyCodeBitree: updated.positionalHierarchyCodeBitree, // For BiTree
            positionalHierarchyDirectionIcicle: updated.positionalHierarchyDirectionIcicle, // For Icicle plot
            justReclusterd: false,
            contributions: updated.contributions,
            magmin: updated.magmin,
            magmax: updated.magmax
        })
    }

    async verticalIcicleListenerBarClick(clickedCodeNodes) {
        /*
            Selection given in the parameter are truncated leaf nodes.

            clickedNodes: name of nodes (intermediate or terminal), and need to be accumulated afterward. The reason
            for doing this instead of passing around leaf node names is to make it easier to control visual highlights.
         */
        // Reset previous selection
        this.setState({selectedCodeNodes: []})
        this.setState({selectedCodeLeaves: []})

        // Gather child node
        clickedCodeNodes = [clickedCodeNodes]

        // Set new selection
        this.setState({selectedCodeNodes: clickedCodeNodes})
        this.setState({selectedCodeLeaves: clickedCodeNodes})

        if (this.state.justReclustered) {
            // Also set variance range
            this.setupVisualization()
            // this.setState({
            //     dirAvgMagnitude: updated.avgMagnitude,
            //     dirAvgStd: updated.avgStd,
            //     magsStd: updated.magsStd,
            //     stdsStd: updated.stdsStd,
            //     positionalHierarchyDirectionBitree: structuredClone(updated.positionalHierarchyDirectionIcicle)
            // })
            this.setState({justReclustered: false})
        } else {
            let updated = await this.verticalIcicleUpdateLogic(clickedCodeNodes)
            this.setState({
                selectedCodeNodes: clickedCodeNodes, // Visual nodes / intermediate nodes
                selectedCodeLeaves: clickedCodeNodes, // Actual leaf nodes
                positionalHierarchyCodeBitree: updated.positionalHierarchyCodeBitree, // For BiTree
                positionalHierarchyDirectionIcicle: updated.positionalHierarchyDirectionIcicle, // For Icicle plot
                justReclustered: false,
                contributions: updated.contributions,
                magmin: updated.magmin,
                magmax: updated.magmax
            })

        }
    }

    async horizontalIcicleUpdateLogic(leafNodeIndices, selectedCodeLeaves) {
        leafNodeIndices = leafNodeIndices.filter(function( element ) { return element !== undefined })

        // Request coherence data for sub-selection
        let response = null
        response = await this.fetchDirectionHierarchySelectionData(this.state.codeHierarchyInitialData, leafNodeIndices, selectedCodeLeaves)

        // Re-compute hierarchyScale of the other tree
        let tmpVScale = hierarchyScale()
            .domain(response.codeTree)
            .range([0, this.state.height - this.state.icicleSize - this.state.toggledBarHeight])
            .paddingInner(0.002)
            .paddingOuter(0.002)
        let positionalHierarchyCodeIcicle = tmpVScale(response.codeTree, true, this.state.visDepth)

        // Data structure for biTree - uses truncated version of the tree.
        let directionTree = this.state.directionHierarchyBitreeData
        // let filteredHierarchy = this.computeFilteredHierarchy(directionTree, leafNodeIndices, 'horizontal')
        let filteredHierarchy = this.computeCulledHierarchy(directionTree, leafNodeIndices, leafNodeIndices)
        let positionalHierarchyDirectionBitree = this.computePositionalHierarchy(filteredHierarchy, 'horizontal')

        // Mark magnitude and variance
        let magnitude = response.magnitude
        let variance = response.variance
        let contributions = response.contributions
        let magmin = response.magmin
        let magmax = response.magmax
        console.log("MagMin MagMax", magmin, magmax)

        return {positionalHierarchyDirectionBitree, positionalHierarchyCodeIcicle, magnitude, variance, contributions, magmin, magmax}
    }

    async horizontalIcicleListenerDuring(selectedDirectionNodes, start, end) {
        /*
        Selection given in the parameter are truncated leaf nodes.
        start, end is the absolution position of the brush.
        */

        // Reset previous selection
        this.setState({selectedDirectionNodes: []})
        this.setState({selectedDirectionLeaves: []})

        // Gather child nodes
        let leafNodeIndices = this.gatherUnderBrushLeaves(selectedDirectionNodes, start, end)

        // Set new selection
        this.setState({selectedDirectionNodes: selectedDirectionNodes})
        this.setState({selectedDirectionLeaves: leafNodeIndices})

        leafNodeIndices = leafNodeIndices.filter(function( element ) { return element !== undefined })

        // Data structure for biTree - uses truncated version of the tree.
        let directionTree = this.state.directionHierarchyBitreeData
        // let filteredHierarchy = this.computeFilteredHierarchy(directionTree, leafNodeIndices, 'horizontal')
        let filteredHierarchy = this.computeCulledHierarchy(directionTree, leafNodeIndices, leafNodeIndices)

        let positionalHierarchyDirectionBitree = this.computePositionalHierarchy(filteredHierarchy, 'horizontal')

        // Positional BiTree need to be modified based on how many leaves from which node are selected under the brush.
        // This information is in leafNodeIndices. Modification can be done in Cull function.

        // Selection
        this.setState({
            selectedDirectionNodes: selectedDirectionNodes, // Selection
            selectedDirectionLeaves: leafNodeIndices,
            positionalHierarchyDirectionBitree: positionalHierarchyDirectionBitree, // BiTree
        })
    }

    async horizontalIcicleListener(selectedDirectionNodes, start, end) { // console.log("Action from horizontal icicle plot")
        /*
            Selection given in the parameter are truncated leaf nodes.
            start, end is the absolution position of the brush.
         */

        // Reset previous selection
        this.setState({selectedDirectionNodes: []})
        this.setState({selectedDirectionLeaves: []})

        // Gather child nodes
        let leafNodeIndices = this.gatherUnderBrushLeaves(selectedDirectionNodes, start, end)

        // Set new selection
        this.setState({selectedDirectionNodes: selectedDirectionNodes})
        this.setState({selectedDirectionLeaves: leafNodeIndices})

        let updated = await this.horizontalIcicleUpdateLogic(leafNodeIndices, this.state.selectedCodeLeaves)

        // Positional BiTree need to be modified based on how many leaves from which node are selected under the brush.
        // This information is in leafNodeIndices. Modification can be done in Cull function.

        // Selection
        this.setState({
            selectedDirectionNodes: selectedDirectionNodes, // Selection
            selectedDirectionLeaves: leafNodeIndices,
            positionalHierarchyDirectionBitree: updated.positionalHierarchyDirectionBitree, // BiTree
            positionalHierarchyCodeIcicle: updated.positionalHierarchyCodeIcicle, // Icicle Plot
            movingMagnitude: updated.magnitude,
            movingVariance: updated.variance,
            contributions: updated.contributions,
            magmin: updated.magmin,
            magmax: updated.magmax
        })
    }


    async horizontalIcicleListenerClick(selectedDirectionNodes) {
        /*
            Selection given in the parameter are truncated leaf nodes.
         */

        // Reset previous selection
        this.setState({selectedDirectionNodes: []})
        this.setState({selectedDirectionLeaves: []})

        // Gather child nodes' name
        let leafNodeIndices = this.gatherUnderNodeLeaves(selectedDirectionNodes)

        // Set new selection
        this.setState({selectedDirectionNodes: selectedDirectionNodes})
        this.setState({selectedDirectionLeaves: leafNodeIndices})

        let updated = await this.horizontalIcicleUpdateLogic(leafNodeIndices, this.state.selectedCodeLeaves)

        // Selection
        this.setState({
            selectedDirectionNodes: selectedDirectionNodes, // Selection
            selectedDirectionLeaves: leafNodeIndices,
            positionalHierarchyDirectionBitree: updated.positionalHierarchyDirectionBitree, // BiTree
            positionalHierarchyCodeIcicle: updated.positionalHierarchyCodeIcicle, // Icicle Plot
            contributions: updated.contributions,
            magmin: updated.magmin,
            magmax: updated.magmax
        })
    }

    async horizontalIcicleListenerBarClick(clickedChildNode) {
        // Reset previous selection
        this.setState({selectedDirectionNodes: []})
        this.setState({selectedDirectionLeaves: []})

        clickedChildNode = [clickedChildNode]

        // Set new selection
        this.setState({selectedDirectionNodes: clickedChildNode}) // TODO: This actually has to be its parent.
        this.setState({selectedDirectionLeaves: clickedChildNode})

        let updated = await this.horizontalIcicleUpdateLogic(clickedChildNode, this.state.selectedCodeLeaves)

        // Selection
        this.setState({
            selectedDirectionNodes: clickedChildNode, // Selection
            selectionDirectionLeaves: clickedChildNode,
            positionalHierarchyDirectionBitree: updated.positionalHierarchyDirectionBitree, // BiTree
            positionalHierarchyCodeIcicle: updated.positionalHierarchyCodeIcicle, // Icicle Plot
            contributions: updated.contributions,
            magmin: updated.magmin,
            magmax: updated.magmax
        })
    }

    async reclusterButtonClickListener(event) {
        /*
        Re-clustering function updates both the clustering and coherence of the direction hierarchy. It will make
        changes to the "initialized data," but will
         */
        // Clear direction selection
        this.clearSelectionDirection()

        // Leaves
        let newChildNodes = this.state.selectedCodeLeaves

        // Data structure for biTree
        // let filteredHierarchy = this.computeFilteredHierarchy(newChildNodes, 'vertical')
        // let positionalHierarchyCodeBitree = this.computePositionalHierarchy(filteredHierarchy, 'vertical')

        // Request coherence data for sub-selection - new tree & new coherency score.
        let directionData = await this.fetchDirectionHierarchyRecluster(
            this.state.directionHierarchyBitreeData, // Need to use truncated one.
            newChildNodes,
            newChildNodes
        )

        let newHorizontalHierarchyScale = hierarchyScale()
            .domain(directionData.directionTree)
            .range([0, this.state.width - this.state.icicleSize - this.state.settingWidth])
            .paddingInner(0.000)
            .paddingOuter(0.000)

        let positionalHierarchyDirectionIcicle = newHorizontalHierarchyScale(directionData.directionTree, true, this.state.visDepth)

        // Weave score min/max
        function _getWeaveScores(node, weaveScores) {
            if (node.leaf) {
                return weaveScores.push(node.weaveScore)
            }
            for (let child of node.children) {
                _getWeaveScores(child, weaveScores)
            }
        }
        let weaveScores = []
        _getWeaveScores(directionData.directionTree, weaveScores)
        // let weaveMin = Math.min(...directionData.weaveScores)
        let weaveMax = Math.max(...weaveScores)

        console.log(directionData)
        this.setState({
            weaveMax: weaveMax,
            directionHierarchyInitialData: directionData.directionRawTree,
            directionHierarchyBitreeData: directionData.directionTree,
            positionalHierarchyDirectionIcicle: positionalHierarchyDirectionIcicle,
            positionalHierarchyDirectionBitree: positionalHierarchyDirectionIcicle,
            justReclustered: true,
            // Enable to update color range when re-clustering
            // dirAvgMagnitude: directionData.avgMagnitude,
            // dirAvgStd: directionData.avgStd,
            // magsStd: directionData.magsStd,
            // stdsStd: directionData.stdsStd
        })
    }

    toggledBarClickListener(event) {
        // Upon clicking, retracts.
        if (this.state.toggledBarHeight === 15)
            this.setState({toggledBarSize: 8})
        else
            this.setState({toggledBarSize: 15})
    }

    async resetButtonClickListener(event) {
        // Reset selections
        console.log("Action from reset button")
        this.clearSelectionCode()
        this.clearSelectionDirection()

        this.setupVisualization()
    }

    mSliderListener(value) {
        this.setState({m_std_interval: value})
    }

    uSliderListener(value) {
        this.setState({u_std_interval: value})
    }

    imageSizeSliderListener(value) {
        this.setState({imageSize: value})
    }

    // Data requests
    async fetchCodeHierarchyInitialization(directions, codeChildren, previous_tree=null) {
        // Load data
        let data = fetch(`http://127.0.0.1:${this.state.port}/conceptlens/code_initialization`, {
            method: 'POST',
            body: JSON.stringify({
                experiment_names: this.state.experimentNames,
                tree: previous_tree,
                directions: directions,
                codes: codeChildren,
                metric: this.state.pairwiseMetric,
                pairwise_metric: this.state.pairwiseMetric,
                clustering_method: this.state.clusteringMethod,
                truncated_tree: this.state.truncatedTree
            })
        }).then(
            response => {
                return response.json()
            }
        ).then(
            data => JSON.parse(data)
        ).then(
            data => {
                let tree = data.tree
                let rawtree = data.rawtree
                let avgMagnitude = data.avgMagnitude
                let avgStd = data.avgStd
                return {tree, rawtree, avgMagnitude, avgStd}
            }
        )
        return data
    }

    async fetchCodeHierarchySelectionData(directionTree, codeIndices, directionIndices) {
        console.log("Passed on to server: ", directionTree) // This is a truncated tree.
        let data = fetch(`http://127.0.0.1:${this.state.port}/conceptlens/code_selection`, {
            method: 'POST',
            body: JSON.stringify({
                experiment_names: this.state.experimentNames,
                direction_tree: directionTree,
                code_indices: codeIndices,
                direction_indices: directionIndices
            })
        }).then(response => response.json())
            .then(data => JSON.parse(data))
            .then(data => {
                console.log("Direction Tree: ", data.directionTree)
                let directionTree = data.directionTree
                let contributions = data.contributions
                let magmin = data.magmin
                let magmax = data.magmax
                return {directionTree, contributions, magmin, magmax}
            })
        return data
    }

    async fetchDirectionHierarchyInitialization(code_selection_clustering=null,
                                                code_selection_coherence=null,
                                                previous_tree=null) {
        // Load data
        // let data = fetch(`http://127.0.0.1:${this.state.port}/conceptlens/direction_hierarchical_clustering`, {
        let data = fetch(`http://127.0.0.1:${this.state.port}/conceptlens/direction_initialization`, {
            method: 'POST',
            body: JSON.stringify({
                experiment_names: this.state.experimentNames,
                tree: previous_tree,
                code_selection_clustering: code_selection_clustering,
                code_selection_coherence: code_selection_coherence,
                feature_processing: this.state.featureProcessingMethod,
                pairwise_metric: this.state.pairwiseMetric,
                clustering_method: this.state.clusteringMethod,
                truncated_tree: this.state.truncatedTree
            })
        }).then(
            response => response.json()
        ).then(
            data => JSON.parse(data)
        ).then(
            data => {
                let tree = data.tree
                let rawtree = data.rawtree
                // console.log("DirectionTree: ", tree)

                // Compute minimum and maximum of weaveScores
                let weaveScores = []
                gatherWeaveScores(tree, weaveScores)

                let avgMagnitude = data.avgMagnitude
                let avgStd = data.avgStd
                let magsStd = data.magsStd
                let stdsStd = data.stdsStd
                return {
                    tree,
                    rawtree,
                    avgMagnitude,
                    avgStd,
                    weaveScores,
                    magsStd,
                    stdsStd
                }
            }
        )
        return data
    }

    async fetchDirectionHierarchySelectionData(codeTree, directionIndices, codeIndices) {
        let data = fetch(`http://127.0.0.1:${this.state.port}/conceptlens/direction_selection`, {
            method: 'POST',
            body: JSON.stringify({
                experiment_names: this.state.experimentNames,
                code_tree: codeTree,
                direction_indices: directionIndices,
                code_indices: codeIndices,
            })
        }).then(response => response.json())
            .then(data => JSON.parse(data))
            .then(data => {
                console.log(data)
                console.log("Code tree: ", data.codeTree)
                let codeTree = data.codeTree
                let magnitude = data.magnitude
                let variance = data.variance
                let contributions = data.contributions
                let magmin = data.magmin
                let magmax = data.magmax
                return {codeTree, magnitude, variance, contributions, magmin, magmax}
            })
        return data
    }

    async fetchDirectionHierarchyRecluster(directionTree, codeIndicesClustering, codeIndicesCoherence) {
        let data = fetch(`http://127.0.0.1:${this.state.port}/conceptlens/direction_recluster` , {
            method: 'POST',
            body: JSON.stringify({
                experiment_names: this.state.experimentNames,
                direction_tree: directionTree,
                code_indices_clustering: codeIndicesClustering,
                code_indices_coherence: codeIndicesCoherence,
                pairwise_metric: this.state.pairwiseMetric,
                clustering_method: this.state.clusteringMethod,
            })
        }).then(response => response.json())
            .then(data => JSON.parse(data))
            .then(data => {
                let directionTree = data.tree
                let directionRawTree = data.rawtree
                let weaveScores = data.weaveScores
                let avgMagnitude = data.avgMagnitude
                let avgStd = data.avgStd
                let magsStd = data.magsStd
                let stdsStd = data.stdsStd
                return {directionTree, directionRawTree, weaveScores, avgMagnitude, avgStd, magsStd, stdsStd}
            })
        return data
    }

    async componentDidUpdate(prevProps, prevState, snapshot) {
        // To avoid an infinite loop, all the network requests are needed to be inside a conditional statement as:
        if (prevState.experimentNames !== this.state.experimentNames ||
            prevState.featureProcessingMethod !== this.state.featureProcessingMethod ||
            prevState.clusteringMethod !== this.state.clusteringMethod ||
            prevState.pairwiseMetric !== this.state.pairwiseMetric ||
            prevState.truncatedTree !== this.state.truncatedTree ||
            prevState.visDepth !== this.state.visDepth
        ) {
            this.setupVisualization()
        }
    }

    async setupVisualization() {
        // Initial data at experiment name selection.
        let codeData = await this.fetchCodeHierarchyInitialization([])
        let directionData = await this.fetchDirectionHierarchyInitialization([])

        // Clone
        let clonedCodeData = structuredClone(codeData)
        let clonedDirectionData = structuredClone(directionData)

        // Hierarchy scale
        let verticalHierarchyScale = hierarchyScale()
            .domain(clonedCodeData.tree)
            .range([0, this.state.height - this.state.icicleSize - this.state.toggledBarHeight])
            .paddingInner(0.001)
            .paddingOuter(0.002)

        let horizontalHierarchyScale = hierarchyScale()
            .domain(clonedDirectionData.tree)
            .range([0, this.state.width - this.state.icicleSize - this.state.settingWidth])
            .paddingInner(0.000)
            .paddingOuter(0.000)

        let positionalHierarchyCode = verticalHierarchyScale(clonedCodeData.tree, true, this.state.visDepth)
        let positionalHierarchyDirection = horizontalHierarchyScale(clonedDirectionData.tree, true, this.state.visDepth)

        // Initial data that will not change before and new experiment selection is made.
        this.setState({
            codeHierarchyInitialData: codeData.rawtree, // Original raw data - untouched, no truncation
            directionHierarchyInitialData: directionData.rawtree,
            codeHierarchyBitreeData: codeData.tree, // Still original data - truncated
            directionHierarchyBitreeData: directionData.tree,
            positionalHierarchyCodeBitree: positionalHierarchyCode, // Bi-hierarchy
            positionalHierarchyDirectionBitree: positionalHierarchyDirection,
            positionalHierarchyCodeIcicle: positionalHierarchyCode, // For icicle plots
            positionalHierarchyDirectionIcicle: positionalHierarchyDirection
        })

        this.setState({verticalHierarchyScale: verticalHierarchyScale})
        this.setState({horizontalHierarchyScale: horizontalHierarchyScale})

        // Measures
        this.setState({codeAvgMagnitude: clonedCodeData.avgMagnitude})
        this.setState({codeAvgStd: clonedCodeData.avgStd})
        this.setState({dirAvgMagnitude: clonedDirectionData.avgMagnitude})
        this.setState({dirAvgStd: clonedDirectionData.avgStd})

        this.setState({magsStd: clonedDirectionData.magsStd})
        this.setState({stdsStd: clonedDirectionData.stdsStd})
    }

    render() {
        console.log("========== Main Render Function Called ==========")
        // Compute element size
        let verticalIciclePlotSize = [this.state.icicleSize, this.state.height - this.state.icicleSize - this.state.toggledBarHeight]
        let horizontalIciclePlotSize = [this.state.width - this.state.icicleSize - this.state.settingWidth, this.state.icicleSize]
        let biTreeSize = [this.state.width - this.state.icicleSize - this.state.settingWidth, this.state.height - this.state.icicleSize - this.state.toggledBarHeight]
        let toggledBarSize = [this.state.width - this.state.icicleSize - this.state.settingWidth, this.state.toggledBarHeight]
        let originalImagePanelSize = [this.state.originalImagePlotSize, this.state.height - this.state.icicleSize - this.state.toggledBarHeight]

        // Translations
        const mainVisTranslation = [60, 60]
        let verticalIciclePlotTranslate = [this.state.settingWidth + mainVisTranslation[0], verticalIciclePlotSize[0] + toggledBarSize[1] + mainVisTranslation[1]]
        const horizontalIciclePlotTranslate = [this.state.settingWidth + horizontalIciclePlotSize[1] + mainVisTranslation[0], mainVisTranslation[1]]
        let biTreeTranslate = [this.state.settingWidth + verticalIciclePlotSize[0] + mainVisTranslation[0], horizontalIciclePlotSize[1] + toggledBarSize[1] + mainVisTranslation[1]]
        const toggledBarTranslate = [this.state.settingWidth + verticalIciclePlotSize[0] + mainVisTranslation[0], horizontalIciclePlotSize[1] + mainVisTranslation[1]]
        let originalImagePanelTranslate = [this.state.settingWidth + this.state.width + this.state.oriGap + mainVisTranslation[0],
            this.state.icicleSize + this.state.toggledBarHeight + mainVisTranslation[1]]

        // Consistency colorScale
        let magSpacing = 1.
        let stdSpacing = 1.

        let magnitudeFloor = this.state.dirAvgMagnitude - magSpacing * this.state.magsStd
        let magnitudeCeil = this.state.dirAvgMagnitude + magSpacing * this.state.magsStd
        let stdFloor = this.state.dirAvgStd - stdSpacing * this.state.stdsStd
        let stdCeil = this.state.dirAvgStd + stdSpacing * this.state.stdsStd
        // let stdFloor = 0
        // let stdCeil = 1

        // VSUP
        let quantization = vsup.quantization(2, 4)
            .valueDomain([
                // this.state.dirAvgMagnitude - (magSpacing * this.state.u_std_interval) < 0 ? 0 : this.state.dirAvgMagnitude - (magSpacing * this.state.u_std_interval),
                // this.state.dirAvgMagnitude + (magSpacing * this.state.u_std_interval)])
                magnitudeFloor, magnitudeCeil
            ])
            .uncertaintyDomain([
                // this.state.dirAvgStd - (stdSpacing * this.state.m_std_interval) < 0 ? 0 : this.state.dirAvgStd - (stdSpacing * this.state.m_std_interval),
                // this.state.dirAvgStd + (stdSpacing * this.state.m_std_interval)
                stdFloor, stdCeil
            ])

        var consistencyColorScale = vsup.scale().quantize(quantization).range(viridis)

        // Define scales for x and y based on the domains and desired pixel ranges
        const vsupxScale = d3.scaleLinear()
            .domain([magnitudeFloor, magnitudeCeil]) // Use the same domain as valueDomain
            .range([0, 110])     // Define range based on your desired size
            .clamp(true)

        const vsupyScale = d3.scaleLinear()
            .domain([stdFloor, stdCeil])             // Use the same domain as uncertaintyDomain
            .range([0, 110])     // Reverse range if you want higher std at the top
            .clamp(true)


        // Angular and radial constraints for the cone
        const angleScale = d3.scaleLinear()
            .domain([magnitudeFloor, magnitudeCeil])  // Domain for `movingMagnitude`
            .range([Math.PI * 0.66, Math.PI * 0.34])  // Angular bounds of the cone
            .clamp(true);  // Clamp to keep within angular bounds

        const radiusScale = d3.scaleLinear()
            .domain([stdFloor, stdCeil])  // Domain for `movingVariance`
            .range([110, 0])  // Radial bounds of the cone
            .clamp(true);  // Clamp to keep within radial bounds

        const createPatterns = () => (
          <defs>
            {/* Diagonal lines (빗금) */}
            <pattern id="diagonal-lines" width="8" height="8" patternUnits="userSpaceOnUse">
              <path d="M0,0 L8,8" stroke="black" strokeWidth="1" />
            </pattern>

            {/* X's */}
            <pattern id="cross-hatch" width="8" height="8" patternUnits="userSpaceOnUse">
              <path d="M0,0 L8,8 M8,0 L0,8" stroke="black" strokeWidth="1" />
            </pattern>

            {/* Solid color for late layers */}
            <pattern id="solid-color" width="10" height="10" patternUnits="userSpaceOnUse">
              <rect width="10" height="10" fill="gray" />
            </pattern>

            {/* Horizontal lines */}
            <pattern id="horizontal-lines" width={1} height={1} patternUnits={"userSpaceOnUse"}>
              <path d={'M0,1 L2,1'} stroke={'black'} strokeWidth={0.8}></path>
            </pattern>

            {/* Vertical lines */}
            <pattern id="vertical-lines" width={1} height={1} patternUnits={"userSpaceOnUse"}>
              <path d={'M1,0 L1,2'} stroke={'black'} strokeWidth={0.8}></path>
            </pattern>
          </defs>
        );

        return (
            <div className={'container'}>
                {/* backgroundColor: "rgba(0, 0, 0, .05)" */}

                <div className={'row'}>
                    <div className={'col-sm-3'}>
                        <div className={'secbox box with-box-shadow'}>
                            <p style={{fontWeight: "bold", fontFamily: "sans-serif", color: d3.hcl(0, 0, 30), textAlign: "center"}}>
                                Controls
                            </p>
                            <div className={'d-grid gap-1'}>
                                <Button variant={'secondary'} size={'sm'} onClick={this.reclusterButtonClickListener}> recluster </Button>
                                <Button variant={'secondary'} size={'sm'} onClick={this.resetButtonClickListener}> reset </Button>
                            </div>
                        </div>
                    </div>
                    {/*<div className={'col-sm-3'}>*/}
                    {/*    <div className={'secbox box with-box-shadow'}>*/}
                    {/*        <p style={{fontWeight: "bold", fontFamily: "sans-serif", color: d3.hcl(0, 0, 30), textAlign: "center"}}>*/}
                    {/*            Interval*/}
                    {/*        </p>*/}
                    {/*        <p> Inconsistency Domain Scaler</p>*/}
                    {/*        <Slider defaultValue={0.4} min={0.05} max={5.0} step={0.01} onChange={this.mSliderListener}/>*/}
                    {/*        <p> Magnitude Domain Scaler</p>*/}
                    {/*        <Slider defaultValue={0.4} min={0.05} max={5.0} step={0.01} onChange={this.uSliderListener}/>*/}
                    {/*    </div>*/}
                    {/*</div>*/}
                    <div className={'col-sm-9'}>
                        <div className={'secbox box with-box-shadow'}>
                            <p style={{fontWeight: "bold", fontFamily: "sans-serif", color: d3.hcl(0, 0, 30), textAlign: "center"}}>
                                Method Color Legend
                            </p>
                            <MethodLegend
                                experimentNames={this.props.experimentNames}
                                methodColorScale={this.state.methodColorScale}
                                size={770}
                                translate={0}
                            />
                        </div>
                    </div>
                    {/*<div className={'col-sm-2'}>*/}
                    {/*    <div className={'secbox box with-box-shadow'}>*/}
                    {/*        <p style={{fontWeight: "bold", fontFamily: "sans-serif", color: d3.hcl(0, 0, 30), textAlign: "center"}}>*/}
                    {/*            Method Comparison*/}
                    {/*        </p>*/}
                    {/*        <MethodComparison*/}
                    {/*            experimentNames={this.props.experimentNames}*/}
                    {/*            methodColorScale={this.state.methodColorScale}*/}
                    {/*            size={200}*/}
                    {/*            translate={0}*/}
                    {/*        />*/}
                    {/*    </div>*/}
                    {/*</div>*/}

                </div>

                <hr/>

                <div className={'row'}>
                    <div className={'col-lg-10'}>
                        <svg ref={this.svgRef}
                             width={this.state.width + this.state.originalImagePlotSize + this.state.oriGap + mainVisTranslation[0] + 5}
                             height={this.state.height + mainVisTranslation[1]}>
                            {createPatterns()}

                            <g width={this.state.width} height={mainVisTranslation[1]}>
                                <text transform={`translate(${verticalIciclePlotSize[0] + (horizontalIciclePlotSize[0] / 2)}, ${mainVisTranslation[1] / 2})`}
                                      textAnchor={'middle'}
                                      fontFamily={'sans-serif'}
                                      fontWeight={'bold'}
                                      fill={d3.hcl(0, 0, 30)}
                                > Concept Direction Hierarchy </text>
                            </g>

                            <g>
                                <LegendPlot
                                    size={110}
                                    tranlate={[0, 0]}
                                    consistencyColorScale={consistencyColorScale}
                                    movingMagnitude={this.state.movingMagnitude}
                                    movingVariance={this.state.movingVariance}
                                    vsupxScale={vsupxScale}
                                    vsupyScale={vsupyScale}
                                    angleScale={angleScale}
                                    radiusScale={radiusScale}
                                />
                            </g>

                            <g width={mainVisTranslation[0]} height={mainVisTranslation[1]}>
                                <text transform={`translate(${mainVisTranslation[0] / 2}, ${mainVisTranslation[0] + horizontalIciclePlotSize[1] + toggledBarSize[1] + (verticalIciclePlotSize[1] / 2)}) rotate(-90)`}
                                      textAnchor={'middle'}
                                      fontFamily={'sans-serif'}
                                      fontWeight={'bold'}
                                      fill={d3.hcl(0, 0, 30)}
                                > Code Hierarchy </text>
                            </g>

                            <g width={15} height={15}>
                                <text transform={`translate(${mainVisTranslation[0] + this.state.width + this.state.oriGap + (this.props.originalImagePlotSize / 2)}, ${mainVisTranslation[1] + horizontalIciclePlotSize[1]})`}
                                      textAnchor={'middle'}
                                      fontFamily={'sans-serif'}
                                      fontWeight={'bold'}
                                      fontSize={'14px'}
                                      fill={d3.hcl(0, 0, 30)}
                                > Original Image </text>
                            </g>

                            <g>
                                <ToggledBar
                                    positionalHierarchyDirection={this.state.positionalHierarchyDirectionIcicle}
                                    methodColorScale={this.state.methodColorScale}
                                    size={toggledBarSize}
                                    translate={toggledBarTranslate}
                                    onclick={this.toggledBarClickListener}
                                />

                                <VerticalIciclePlot
                                    parentG={this.svgRef.current}
                                    positionalHierarchyData={this.state.positionalHierarchyCodeIcicle}
                                    colorScale={consistencyColorScale}
                                    size={verticalIciclePlotSize}
                                    translate={verticalIciclePlotTranslate}
                                    visDepth={this.state.visDepth}
                                    brushListener={this.verticalIcicleListener}
                                    brushListenerDuring={this.verticalIcicleListenerDuring}
                                    clickListener={this.verticalIcicleListenerClick}
                                    clickBarListener={this.verticalIcicleListenerBarClick}
                                    getSelectionCode={this.getSelectionCode}
                                    experimentNames={this.state.experimentNames}
                                    magnitudeFloor={magnitudeFloor}
                                    magnitudeCeil={magnitudeCeil}
                                />

                                <HorizontalIciclePlot
                                    parentG={this.svgRef.current}
                                    positionalHierarchyData={this.state.positionalHierarchyDirectionIcicle}
                                    colorScale={consistencyColorScale}
                                    size={horizontalIciclePlotSize}
                                    translate={horizontalIciclePlotTranslate}
                                    visDepth={this.state.visDepth}
                                    brushListener={this.horizontalIcicleListener}
                                    brushListenerDuring={this.horizontalIcicleListenerDuring}
                                    clickListener={this.horizontalIcicleListenerClick}
                                    clickBarListener={this.horizontalIcicleListenerBarClick}
                                    getSelectionDirection={this.getSelectionDirection}
                                    experimentNames={this.state.experimentNames}
                                    weaveMax={this.state.weaveMax}
                                    magnitudeFloor={magnitudeFloor}
                                    magnitudeCeil={magnitudeCeil}
                                    stdFloor={stdFloor}
                                    stdCeil={stdCeil}
                                />

                                <BiTree
                                    parentG={this.svgRef}
                                    experimentNames={this.state.experimentNames}
                                    positionalHierarchyCode={this.state.positionalHierarchyCodeBitree}
                                    positionalHierarchyDirection={this.state.positionalHierarchyDirectionBitree}
                                    methodColorScale={this.state.methodColorScale}
                                    size={biTreeSize}
                                    imageSize={this.state.imageSize}
                                    translate={biTreeTranslate}
                                    icicleSize={this.state.icicleSize}
                                    visDepth={this.state.visDepth}
                                    port={this.state.port}
                                    contributions={this.state.contributions}
                                    magmin={this.state.magmin}
                                    magmax={this.state.magmax}
                                    codeSelectionOrder={this.state.selectedCodeLeaves}
                                    directionSelectionOrder={this.state.selectedDirectionLeaves}
                                />



                                <OriginalImagePanel
                                    parentG={this.svgRef}
                                    positionalHierarchyCode={this.state.positionalHierarchyCodeBitree}
                                    size={originalImagePanelSize}
                                    translate={originalImagePanelTranslate}
                                    visDepth={this.state.visDepth}
                                    port={this.state.port}
                                    experimentNames={this.state.experimentNames}
                                />
                            </g>
                        </svg>
                    </div>
                </div>
            </div>
        )
    }
}
