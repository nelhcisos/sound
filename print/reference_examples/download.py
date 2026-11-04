import json,sys,os,urllib.request
API="https://api.printables.com/graphql/"
def gql(q,v):
    r=urllib.request.Request(API,json.dumps({"query":q,"variables":v}).encode(),{"content-type":"application/json","user-agent":"Mozilla/5.0"})
    return json.load(urllib.request.urlopen(r))
for pid,folder in [(a.split(":")[0],a.split(":")[1]) for a in sys.argv[1:]]:
    p=gql("query($id:ID!){print(id:$id){name stls{id name} otherFiles{id name}}}",{"id":pid})["data"]["print"]
    os.makedirs("models/"+folder,exist_ok=True)
    for kind,lst in (("stl",p["stls"]),("other",p["otherFiles"] or [])):
        for f in lst:
            d=gql("mutation($id:ID!,$p:ID!,$t:DownloadFileTypeEnum!){getDownloadLink(id:$id,printId:$p,fileType:$t,source:model_detail){ok output{link}}}",{"id":f["id"],"p":pid,"t":kind})
            link=d["data"]["getDownloadLink"]["output"]["link"]
            import subprocess; subprocess.run(["curl","-sSfL","-o","models/%s/%s"%(folder,f["name"]),urllib.request.quote(link,safe=":/%")],check=True)
            print(folder,f["name"])
