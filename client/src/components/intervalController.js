import React, {Component} from "react";
import * as d3 from "d3"

export default class IntervalController extends Component {
    constructor(props) {
        super();
        this.gref = React.createRef()
        this.state = {
            width: null,
            height: null
        }
    }

    static getDerivedStateFromProps(nextProps, prevState) {
        /*
            Whenever props change, this function will be invoked.
         */
        const {
            width, height
        } = nextProps;

        return {
            width, height
        };
    }

    componentDidUpdate(prevProps, prevState, snapshot) {

    }

    shouldComponentUpdate(nextProps, nextState, nextContext) {
    }

    render() {

        return (
            <svg ref={this.gref} width={this.state.width} height={this.state.height}/>
        )
    }
}