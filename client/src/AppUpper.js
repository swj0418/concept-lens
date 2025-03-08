import { Component, createRef } from "react";
import App from "./App";
import $ from 'jquery';

// Style
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import * as d3 from "d3";

const EXPERIMENTS = {
    "CelebA 256": [
        {id: "ldm_celeba256-vac-global-all", name: "LDM CelebA VAC Global"},
        {id: "ldm_celeba256-vac-global-middle_1", name: "LDM CelebA VAC Global Middle 1"},
        {id: "ldm_celeba256-vac-global-late_1", name: "LDM CelebA VAC Global Late 1"},
        {id: "s2_celeba256-vac-global-early_0", name: "CelebA VAC Global Early 0"},
        {id: "s2_celeba256-sefakmc-global-early_0", name: "CelebA SeFA Global Early 0"},
        {id: "s2_celeba256-sefakmc-global-early_1", name: "CelebA SeFA Global Early 1"},
        {id: "s2_celeba256-sefakmc-global-middle_0", name: "CelebA SeFA Global Middle 0"},
        {id: "s2_celeba256-sefakmc-global-middle_1", name: "CelebA SeFA Global Middle 1"},
    ],
    "FFHQ 256 SeFA Global": [
        {id: "s2_ffhq256-sefakmc-global-early_0",  name: "Early 0"},
        {id: "s2_ffhq256-sefakmc-global-early_1",  name: "Early 1"},
        {id: "s2_ffhq256-sefakmc-global-middle_0", name: "Middle 0"},
        {id: "s2_ffhq256-sefakmc-global-middle_1", name: "Middle 1"},
        {id: "s2_ffhq256-sefakmc-global-middle_2", name: "Middle 2"},
        {id: "s2_ffhq256-sefakmc-global-late_0",   name: "Late 0"},
        {id: "s2_ffhq256-sefakmc-global-late_1",   name: "Late 1"},
    ],
    "FFHQ 256 GS Global": [
        {id: "s2_ffhq256-ganspacekmc-global-early_0",  name: "Early 0"},
        {id: "s2_ffhq256-ganspacekmc-global-early_1",  name: "Early 1"},
        {id: "s2_ffhq256-ganspacekmc-global-middle_0", name: "Middle 0"},
        {id: "s2_ffhq256-ganspacekmc-global-middle_1", name: "Middle 1"},
        {id: "s2_ffhq256-ganspacekmc-global-middle_2", name: "Middle 2"},
        {id: "s2_ffhq256-ganspacekmc-global-late_0",   name: "Late 0"},
        {id: "s2_ffhq256-ganspacekmc-global-late_1",   name: "Late 1"},
    ],
    "FFHQ 256 VAC Global": [
        {id: "s2_ffhq256-vac-global-early_0",  name: "Early 0"},
        {id: "s2_ffhq256-vac-global-early_1",  name: "Early 1"},
        {id: "s2_ffhq256-vac-global-middle_0", name: "Middle 0"},
        {id: "s2_ffhq256-vac-global-middle_1", name: "Middle 1"},
        {id: "s2_ffhq256-vac-global-middle_2", name: "Middle 2"},
        {id: "s2_ffhq256-vac-global-late_0",   name: "Late 0"},
        {id: "s2_ffhq256-vac-global-late_1",   name: "Late 1"},
    ],
    "FFHQ 256 SVM Global": [
        {id: "s2_ffhq256-svmw-global-early_0",  name: "Early 0"},
        {id: "s2_ffhq256-svmw-global-early_1",  name: "Early 1"},
        {id: "s2_ffhq256-svmw-global-middle_0", name: "Middle 0"},
        {id: "s2_ffhq256-svmw-global-middle_1", name: "Middle 1"},
        {id: "s2_ffhq256-svmw-global-middle_2", name: "Middle 2"},
        {id: "s2_ffhq256-svmw-global-late_0",   name: "Late 0"},
        {id: "s2_ffhq256-svmw-global-late_1",   name: "Late 1"},
    ],
    "FFHQ 256 SeFA Layerwise": [
        {id: "s2_ffhq256-sefakmc-layerwise-early_0",  name: "Early 0"},
        {id: "s2_ffhq256-sefakmc-layerwise-early_1",  name: "Early 1"},
        {id: "s2_ffhq256-sefakmc-layerwise-middle_0", name: "Middle 0"},
        {id: "s2_ffhq256-sefakmc-layerwise-middle_1", name: "Middle 1"},
        {id: "s2_ffhq256-sefakmc-layerwise-middle_2", name: "Middle 2"},
        {id: "s2_ffhq256-sefakmc-layerwise-late_0",   name: "Late 0"},
        {id: "s2_ffhq256-sefakmc-layerwise-late_1",   name: "Late 1"},
    ],
    "FFHQ 256 GS Layerwise": [
        {id: "s2_ffhq256-ganspacekmc-layerwise-early_0",  name: "Early 0"},
        {id: "s2_ffhq256-ganspacekmc-layerwise-early_1",  name: "Early 1"},
        {id: "s2_ffhq256-ganspacekmc-layerwise-middle_0", name: "Middle 0"},
        {id: "s2_ffhq256-ganspacekmc-layerwise-middle_1", name: "Middle 1"},
        {id: "s2_ffhq256-ganspacekmc-layerwise-middle_2", name: "Middle 2"},
        {id: "s2_ffhq256-ganspacekmc-layerwise-late_0",   name: "Late 0"},
        {id: "s2_ffhq256-ganspacekmc-layerwise-late_1",   name: "Late 1"},
    ],
    "FFHQ 256 VAC Layerwise": [
        {id: "s2_ffhq256-vac-layerwise-early_0",  name: "Early 0"},
        {id: "s2_ffhq256-vac-layerwise-early_1",  name: "Early 1"},
        {id: "s2_ffhq256-vac-layerwise-middle_0", name: "Middle 0"},
        {id: "s2_ffhq256-vac-layerwise-middle_1", name: "Middle 1"},
        {id: "s2_ffhq256-vac-layerwise-middle_2", name: "Middle 2"},
        {id: "s2_ffhq256-vac-layerwise-late_0",   name: "Late 0"},
        {id: "s2_ffhq256-vac-layerwise-late_1",   name: "Late 1"},
    ],
    "FFHQ 256 SVM Layerwise": [
        {id: "s2_ffhq256-svmw-layerwise-early_0",  name: "Early 0"},
        {id: "s2_ffhq256-svmw-layerwise-early_1",  name: "Early 1"},
        {id: "s2_ffhq256-svmw-layerwise-middle_0", name: "Middle 0"},
        {id: "s2_ffhq256-svmw-layerwise-middle_1", name: "Middle 1"},
        {id: "s2_ffhq256-svmw-layerwise-middle_2", name: "Middle 2"},
        {id: "s2_ffhq256-svmw-layerwise-late_0",   name: "Late 0"},
        {id: "s2_ffhq256-svmw-layerwise-late_1",   name: "Late 1"},
    ],
    "FFHQ 256 GS Male/Female Layerwise": [
        {id: "s2_ffhq256-ganspacekmc_male-layerwise-early_0",    name: "Male Early 0"},
        {id: "s2_ffhq256-ganspacekmc_male-layerwise-early_1",    name: "Male Early 1"},
        {id: "s2_ffhq256-ganspacekmc_male-layerwise-middle_0",   name: "Male Middle 0"},
        {id: "s2_ffhq256-ganspacekmc_male-layerwise-middle_1",   name: "Male Middle 1"},
        {id: "s2_ffhq256-ganspacekmc_male-layerwise-middle_2",   name: "Male Middle 2"},
        {id: "s2_ffhq256-ganspacekmc_male-layerwise-late_0",     name: "Male Late 0"},
        {id: "s2_ffhq256-ganspacekmc_male-layerwise-late_1",     name: "Male Late 1"},
        {id: "s2_ffhq256-ganspacekmc_female-layerwise-early_0",  name: "Female Early 0"},
        {id: "s2_ffhq256-ganspacekmc_female-layerwise-early_1",  name: "Female Early 1"},
        {id: "s2_ffhq256-ganspacekmc_female-layerwise-middle_0", name: "Female Middle 0"},
        {id: "s2_ffhq256-ganspacekmc_female-layerwise-middle_1", name: "Female Middle 1"},
        {id: "s2_ffhq256-ganspacekmc_female-layerwise-middle_2", name: "Female Middle 2"},
        {id: "s2_ffhq256-ganspacekmc_female-layerwise-late_0",   name: "Female Late 0"},
        {id: "s2_ffhq256-ganspacekmc_female-layerwise-late_1",   name: "Female Late 1"},
    ],
    "Wild 512 VA Global": [
        {id: "s2_wild512-vac-global-early_0", name:  "FFHQ VAC Global Early 0"},
        {id: "s2_wild512-vac-global-early_1", name:  "FFHQ VAC Global Early 1"},
        {id: "s2_wild512-vac-global-middle_0", name: "FFHQ VAC Global Middle 0"},
        {id: "s2_wild512-vac-global-middle_1", name: "FFHQ VAC Global Middle 1"},
        {id: "s2_wild512-vac-global-middle_2", name: "FFHQ VAC Global Middle 2"},
        {id: "s2_wild512-vac-global-late_0", name:   "FFHQ VAC Global Late 0"},
        {id: "s2_wild512-vac-global-late_1", name:   "FFHQ VAC Global Late 1"},
    ],
    "Wild 512 SeFA Global": [
        {id: "s2_wild512-sefakmc-global-early_0", name:  "FFHQ SeFA Global Early 0"},
        {id: "s2_wild512-sefakmc-global-early_1", name:  "FFHQ SeFA Global Early 1"},
        {id: "s2_wild512-sefakmc-global-middle_0", name: "FFHQ SeFA Global Middle 0"},
        {id: "s2_wild512-sefakmc-global-middle_1", name: "FFHQ SeFA Global Middle 1"},
        {id: "s2_wild512-sefakmc-global-middle_2", name: "FFHQ SeFA Global Middle 2"},
        {id: "s2_wild512-sefakmc-global-late_0", name:   "FFHQ SeFA Global Late 0"},
        {id: "s2_wild512-sefakmc-global-late_1", name:   "FFHQ SeFA Global Late 1"},
    ],
    "Wild 512 GS Global": [
        {id: "s2_wild512-ganspacekmc-global-early_0", name:  "FFHQ GanSpace Global Early 0"},
        {id: "s2_wild512-ganspacekmc-global-early_1", name:  "FFHQ GanSpace Global Early 1"},
        {id: "s2_wild512-ganspacekmc-global-middle_0", name: "FFHQ GanSpace Global Middle 0"},
        {id: "s2_wild512-ganspacekmc-global-middle_1", name: "FFHQ GanSpace Global Middle 1"},
        {id: "s2_wild512-ganspacekmc-global-middle_2", name: "FFHQ GanSpace Global Middle 2"},
        {id: "s2_wild512-ganspacekmc-global-late_0", name:   "FFHQ GanSpace Global Late 0"},
        {id: "s2_wild512-ganspacekmc-global-late_1", name:   "FFHQ GanSpace Global Late 1"},
        {id: "s2_wild512-ganspacekmc-global-late_1", name:   "FFHQ GanSpace Global Late 1"},
    ],
    "Wild 512 Vector Arithmetic Layerwise": [
        {id: "s2_wild512-vac-layerwise-early_0", name:  "FFHQ VAC Layerwise Early 0"},
        {id: "s2_wild512-vac-layerwise-early_1", name:  "FFHQ VAC Layerwise Early 1"},
        {id: "s2_wild512-vac-layerwise-middle_0", name: "FFHQ VAC Layerwise Middle 0"},
        {id: "s2_wild512-vac-layerwise-middle_1", name: "FFHQ VAC Layerwise Middle 1"},
        {id: "s2_wild512-vac-layerwise-middle_2", name: "FFHQ VAC Layerwise Middle 2"},
        {id: "s2_wild512-vac-layerwise-late_0", name:   "FFHQ VAC Layerwise Late 0"},
        {id: "s2_wild512-vac-layerwise-late_1", name:   "FFHQ VAC Layerwise Late 1"},
    ],
    "Landscape 256 SeFA Global": [
        {id: "s3_landscape256-sefakmc-global-early_0", name:  "Scene SeFA Global Early 0"},
        {id: "s3_landscape256-sefakmc-global-early_1", name:  "Scene SeFA Global Early 1"},
        {id: "s3_landscape256-sefakmc-global-middle_0", name:  "Scene SeFA Global Middle 0"},
        {id: "s3_landscape256-sefakmc-global-middle_1", name:  "Scene SeFA Global Middle 1"},
        {id: "s3_landscape256-sefakmc-global-middle_2", name:  "Scene SeFA Global Middle 2"},
    ],
    "Landscape 256 GS Global": [
        {id: "s3_landscape256-ganspacekmc-global-early_0", name:   "Scene GANSpace Global Early 0"},
        {id: "s3_landscape256-ganspacekmc-global-early_1", name:   "Scene GANSpace Global Early 1"},
        {id: "s3_landscape256-ganspacekmc-global-middle_0", name:  "Scene GANSpace Global Middle 0"},
        {id: "s3_landscape256-ganspacekmc-global-middle_1", name:  "Scene GANSpace Global Middle 1"},
        {id: "s3_landscape256-ganspacekmc-global-middle_2", name:  "Scene GANSpace Global Middle 2"},
    ],
    "Landscape 256 VAC Global": [
        {id: "s3_landscape256-vac-global-early_0", name:   "Scene VA Global Early 0"},
        {id: "s3_landscape256-vac-global-early_1", name:   "Scene VA Global Early 1"},
        {id: "s3_landscape256-vac-global-middle_0", name:  "Scene VA Global Middle 0"},
        {id: "s3_landscape256-vac-global-middle_1", name:  "Scene VA Global Middle 1"},
        {id: "s3_landscape256-vac-global-middle_2", name:  "Scene VA Global Middle 2"},
    ],
    "Landscape 256 SVM Global": [
        {id: "s3_landscape256-svmw-global-early_0", name:   "Scene SVM Global Early 0"},
        {id: "s3_landscape256-svmw-global-early_1", name:   "Scene SVM Global Early 1"},
        {id: "s3_landscape256-svmw-global-middle_0", name:  "Scene SVM Global Middle 0"},
        {id: "s3_landscape256-svmw-global-middle_1", name:  "Scene SVM Global Middle 1"},
        {id: "s3_landscape256-svmw-global-middle_2", name:  "Scene SVM Global Middle 2"},
    ],
    "Landscape 256 SeFA Layerwise": [
        {id: "s3_landscape256-sefakmc-layerwise-early_0", name:   "Scene SeFA Layerwise Early 0"},
        {id: "s3_landscape256-sefakmc-layerwise-early_1", name:   "Scene SeFA Layerwise Early 1"},
        {id: "s3_landscape256-sefakmc-layerwise-middle_0", name:  "Scene SeFA Layerwise Middle 0"},
        {id: "s3_landscape256-sefakmc-layerwise-middle_1", name:  "Scene SeFA Layerwise Middle 1"},
        {id: "s3_landscape256-sefakmc-layerwise-middle_2", name:  "Scene SeFA Layerwise Middle 2"},
    ],
    "Landscape 256 GS Layerwise": [
        {id: "s3_landscape256-ganspacekmc-layerwise-early_0", name:   "Scene GANSpace Layerwise Early 0"},
        {id: "s3_landscape256-ganspacekmc-layerwise-early_1", name:   "Scene GANSpace Layerwise Early 1"},
        {id: "s3_landscape256-ganspacekmc-layerwise-middle_0", name:  "Scene GANSpace Layerwise Middle 0"},
        {id: "s3_landscape256-ganspacekmc-layerwise-middle_1", name:  "Scene GANSpace Layerwise Middle 1"},
        {id: "s3_landscape256-ganspacekmc-layerwise-middle_2", name:  "Scene GANSpace Layerwise Middle 2"},
    ],
    "Landscape 256 VAC Layerwise": [
        {id: "s3_landscape256-vac-layerwise-early_0", name:   "Scene VA Layerwise Early 0"},
        {id: "s3_landscape256-vac-layerwise-early_1", name:   "Scene VA Layerwise Early 1"},
        {id: "s3_landscape256-vac-layerwise-middle_0", name:  "Scene VA Layerwise Middle 0"},
        {id: "s3_landscape256-vac-layerwise-middle_1", name:  "Scene VA Layerwise Middle 1"},
        {id: "s3_landscape256-vac-layerwise-middle_2", name:  "Scene VA Layerwise Middle 2"},
    ],
    "Landscape 256 SVM Layerwise": [
        {id: "s3_landscape256-svmw-layerwise-early_0", name:   "Scene SVM Layerwise Early 0"},
        {id: "s3_landscape256-svmw-layerwise-early_1", name:   "Scene SVM Layerwise Early 1"},
        {id: "s3_landscape256-svmw-layerwise-middle_0", name:  "Scene SVM Layerwise Middle 0"},
        {id: "s3_landscape256-svmw-layerwise-middle_1", name:  "Scene SVM Layerwise Middle 1"},
        {id: "s3_landscape256-svmw-layerwise-middle_2", name:  "Scene SVM Layerwise Middle 2"},
    ],
};

export default class AppUpper extends Component {
    constructor() {
        super();
        this.svgRef = createRef();
        this.state = {
            availableExperiments: {},  // Will be populated by API
            experimentNames: [],
            methodBlindMode: false,
            processingMethod: 'end',
            clusteringMethod: 'complete',
            pairwiseMetric: 'cosine',
            height: 960,
            width: 1200,
            icicleSize: 160,
            settingWidth: 0,
            toggledBarSize: 12,
            originalImagePlotSize: 150,
            imageSize: 100,
            oriGap: 10,
            visDepth: 7,
            truncatedTree: true,
            port: 37203
        };

        // Meta listeners for brush manipulation
        document.addEventListener('keydown', function (event) {
            if (event.altKey) {
                d3.selectAll('.brush').raise();
            }
        });

        document.addEventListener('keydown', function (event) {
            if (event.ctrlKey) {
                d3.selectAll('.brush').lower();
            }
        });

        this.requestAvailableExperiments = this.requestAvailableExperiments.bind(this);
        this.requestAvailableExperiments();
        this.checkboxOnChange = this.checkboxOnChange.bind(this);
        this.methodBlindModeOnChange = this.methodBlindModeOnChange.bind(this);
        this.processingMethodOnChange = this.processingMethodOnChange.bind(this);
        this.clusteringMethodOnChange = this.clusteringMethodOnChange.bind(this);
        this.pairwiseMetricOnChange = this.pairwiseMetricOnChange.bind(this);
        this.visDepthOnChange = this.visDepthOnChange.bind(this);
        this.truncatedTreeOnChange = this.truncatedTreeOnChange.bind(this);
    }

    checkboxOnChange(e) {
        let newExperimentNames = [...this.state.experimentNames];
        if (newExperimentNames.includes(e.target.value))
            this.removeItemOnce(newExperimentNames, e.target.value);
        else
            newExperimentNames.push(e.target.value);

        this.setState({ experimentNames: newExperimentNames });
    }

    async requestAvailableExperiments() {
        try {
            let response = await fetch(`http://127.0.0.1:${this.state.port}/conceptlens/available_experiments`, {
                method: 'POST',
                body: JSON.stringify({})
            });
            let data = await response.json();
            // Assuming data is an object with experiment groups as keys
            this.setState({ availableExperiments: data });
        } catch (error) {
            console.error('Error fetching experiments:', error);
        }
    }

    removeItemOnce(arr, value) {
        const index = arr.indexOf(value);
        if (index > -1) {
            arr.splice(index, 1);
        }
        return arr;
    }

    methodBlindModeOnChange(e) { this.setState({ methodBlindMode: e.target.value }); }
    processingMethodOnChange(e) { this.setState({ processingMethod: e.target.value }); }
    clusteringMethodOnChange(e) { this.setState({ clusteringMethod: e.target.value }); }
    pairwiseMetricOnChange(e) { this.setState({ pairwiseMetric: e.target.value }); }
    visDepthOnChange(e) { this.setState({ visDepth: parseInt(e.target.value) }); }
    truncatedTreeOnChange(e) { this.setState({ truncatedTree: e.target.value }); }

    renderExperimentCheckboxes() {
        const experiments = this.state.availableExperiments;
        // If experiments data hasn't been fetched yet, show a loading message
        if (!experiments || Object.keys(experiments).length === 0) {
            return <div>Loading experiments...</div>;
        }
        // Iterate over each experiment group and render checkboxes
        return Object.keys(experiments).map(section => (
            <div key={section}>
                <h5>{section}</h5>
                {experiments[section].map(exp => (
                    <Row key={exp.id}>
                        <Col xs={9}>{exp.name}</Col>
                        <Col>
                            <input type="checkbox" value={exp.id} onChange={this.checkboxOnChange} />
                        </Col>
                    </Row>
                ))}
            </div>
        ));
    }

    render() {
        const toggledBarHeight = this.state.toggledBarSize;
        return (
            <Container fluid>
                <br /><br />
                <Row>
                    <Col xs={2}>
                        <Row className={'row'}>
                            <h4> Methods </h4>
                            {this.renderExperimentCheckboxes()}
                        </Row>

                        <hr />

                        <Row onChange={this.processingMethodOnChange}>
                            <h5> Feature Processing Method Selection </h5>
                            <Row>
                                <Col> Vector Difference </Col>
                                <Col>
                                    <input type="radio" value="diff" name="pmd" />
                                </Col>
                            </Row>
                            <Row>
                                <Col> Vector End </Col>
                                <Col>
                                    <input type="radio" value="end" name="pmd" />
                                </Col>
                            </Row>
                        </Row>
                        <Row onChange={this.clusteringMethodOnChange}>
                            <h5> Clustering Method </h5>
                            <Row>
                                <Col> Complete </Col>
                                <Col>
                                    <input type="radio" value="complete" name="cm" />
                                </Col>
                            </Row>
                            <Row>
                                <Col> Ward </Col>
                                <Col>
                                    <input type="radio" value="ward" name="cm" />
                                </Col>
                            </Row>
                        </Row>
                        <Row onChange={this.pairwiseMetricOnChange}>
                            <h5> Pairwise Distance Metric </h5>
                            <Row>
                                <Col> Cosine </Col>
                                <Col>
                                    <input type="radio" value="cosine" name="pdm" />
                                </Col>
                            </Row>
                            <Row>
                                <Col> Euclidean </Col>
                                <Col>
                                    <input type="radio" value="euclidean" name="pdm" />
                                </Col>
                            </Row>
                            <Row>
                                <Col> Raw </Col>
                                <Col>
                                    <input type="radio" value="raw" name="pdm" />
                                </Col>
                            </Row>
                        </Row>
                        <Row onChange={this.truncatedTreeOnChange}>
                            <h5> Tree Truncation </h5>
                            <Row>
                                <Col> True </Col>
                                <Col>
                                    <input type="radio" value="True" name="tt" />
                                </Col>
                            </Row>
                            <Row>
                                <Col> False </Col>
                                <Col>
                                    <input type="radio" value="False" name="tt" />
                                </Col>
                            </Row>
                        </Row>
                        <Row onChange={this.visDepthOnChange}>
                            <h5> Tree Visualization Depth </h5>
                            <Row>
                                <Col>
                                    <input type="range" min="1" max="12" step="1" className="slider" />
                                </Col>
                            </Row>
                        </Row>
                    </Col>
                    <Col lg={10}>
                        <App
                            experimentNames={this.state.experimentNames}
                            methodBlindMode={this.state.methodBlindMode}
                            featureProcessingMethod={this.state.processingMethod}
                            clusteringMethod={this.state.clusteringMethod}
                            pairwiseMetric={this.state.pairwiseMetric}
                            truncatedTree={this.state.truncatedTree}
                            visDepth={this.state.visDepth}
                            height={this.state.height}
                            width={this.state.width}
                            icicleSize={this.state.icicleSize}
                            originalImagePlotSize={this.state.originalImagePlotSize}
                            imageSize={this.state.imageSize}
                            toggledBarHeight={toggledBarHeight}
                        />
                    </Col>
                </Row>
            </Container>
        );
    }
}