export function splitExperimentName(expName) {
    const splits = expName.split("-")
    const domainName = splits[0]
    const methodName = splits[1]
    const applicationName = splits[2]
    const layerName = splits[3].split("_")[0]
    const layerSubName = splits[3].split("_")[1]

    return [domainName, methodName, applicationName, layerName, layerSubName]
}