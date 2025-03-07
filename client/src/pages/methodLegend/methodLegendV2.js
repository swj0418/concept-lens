import React, {Component} from "react";
import * as d3 from "d3"

// import { useEffect, useMemo, useState } from "react";
// import { useTable, useSortBy, useRowState } from "react-table";

// import { makeData } from "./data";
// import { Table, Header, Details, Cell } from "./style";
// import { DatesSection, DatesLabels, DatesCell, findBounds } from "./dates";

// export default () => {
//     const [dateBounds, setDateBounds] = useState([]);
//
//     const columns = useMemo(
//         () => [
//             {
//                 Header: "Layer",
//                 accessor: "layer"
//             },
//             {
//                 Header: () => <DatesLabels bounds={dateBounds} />,
//                 accessor: "dates",
//                 Cell: ({ cell, row }) => (
//                     <DatesCell bounds={dateBounds} value={cell.value} />
//                 )
//             }
//         ],
//         [dateBounds]
//     );
//
//     const data = useMemo(makeData, []);
//
//     useEffect(() => setDateBounds(findBounds(data)), [data]);
//
//     const { getTableProps, flatHeaders, rows, prepareRow } = useTable(
//         { columns, data },
//         useSortBy,
//         useRowState
//     );
//
//     return (
//         <Table {...getTableProps()}>
//             <Details>
//                 {flatHeaders.reduce(
//                     (columns, column) =>
//                         column.id !== "dates"
//                             ? [
//                                 ...columns,
//                                 <Header
//                                     {...column.getHeaderProps(column.getSortByToggleProps())}
//                                 >
//                                     {column.render("Header")}
//                                     {column.isSorted
//                                         ? column.isSortedDesc
//                                             ? " 🔽"
//                                             : " 🔼"
//                                         : ""}
//                                 </Header>
//                             ]
//                             : columns,
//                     []
//                 )}
//                 {rows.map(row => {
//                     prepareRow(row);
//                     return row.cells.reduce(
//                         (cells, cell) =>
//                             cell.column.id !== "dates"
//                                 ? [
//                                     ...cells,
//                                     <Cell
//                                         {...cell.getCellProps()}
//                                         rowHovered={row.state.hovered}
//                                         onMouseEnter={() => row.setState({ hovered: true })}
//                                         onMouseLeave={() => row.setState({ hovered: false })}
//                                     >
//                                         {cell.render("Cell")}
//                                     </Cell>
//                                 ]
//                                 : cells,
//                         []
//                     );
//                 })}
//             </Details>
//
//             <DatesSection headers={flatHeaders} prepareRow={prepareRow} rows={rows} />
//         </Table>
//     );
// };


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
        const  {methodNames, size, translate} = nextProps;

        let methodColorScale = d3.scaleSequential(d3.interpolateRainbow).domain([0, 2])

        return {methodNames, methodColorScale, size, translate};
    }

    componentDidUpdate(prevProps, prevState, snapshot) {

    }

    shouldComponentUpdate(nextProps, nextState, nextContext) {
        if (nextState.methodNames)
            return true
        else if (nextState.methodNames !== this.state.methodNames) {
            return true
        }
    }

    render() {
        let g = d3.select(this.gref.current)
        g.selectAll('g').remove()

        const rowItemLimit = 4
        const nRow = Math.floor(this.state.methodNames.length / 5)

        const heightAllocation = 30 * nRow

        let xScale = d3.scaleBand().domain([0, rowItemLimit]).range([150, this.state.size - 250])
        let yScale = d3.scaleLinear().domain([0, nRow]).range([0, heightAllocation])

        let cont = g.selectAll()
            .data(this.state.methodNames)
            .enter()
            .append('g')
            .attr('transform', (d, i) => {
                let rowIdx = Math.floor(i / rowItemLimit)
                let colIdx = i % rowItemLimit

                return `translate(${xScale(colIdx)}, ${yScale(rowIdx)})`
            })

        cont.append('rect')
            .attr('fill', (d, i) => this.state.methodColorScale(i))
            .attr('width', 60)
            .attr('height', 20)

        // cont.append('text')
        //     .text(d => d)
        //     .attr('text-anchor', 'start')
        //     .attr('line-height', 100)
        //     .attr('text-align', 'center')
        //     .attr('font-size', '10px')
        //     .attr('transform', `translate(${60 + 10}, ${20 / 2 + 3})`)

        return (
            <svg ref={this.gref} width={this.state.size} height={100}/>
        )
    }
}