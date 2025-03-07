import React, {Component} from "react";
import * as d3 from "d3"


export default class SettingView extends Component {
    constructor(props) {
        super();
        this.state = {

        }
    }

    static getDerivedStateFromProps(nextProps, prevState) {
        /*
            Whenever props change, this function will be invoked.
         */
        const {} = nextProps;

        return {};
    }

    componentDidUpdate(prevProps, prevState, snapshot) {

    }

    render() {
        return (
            <div>
                <p> Experiments </p>
            </div>
        )
    }
}