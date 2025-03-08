export function splitExperimentName(expName) {
   const splits = expName.split("-")
   const domainName = splits[0]
   const methodName = splits[1]


   const layerName = splits[2]
   // const layerSubName = splits[3].split("_")[1]
   const layerSubName = null


   const applicationName = splits[3]


   return [domainName, methodName, applicationName, layerName, layerSubName]
}
