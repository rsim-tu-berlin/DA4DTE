import org.semanticweb.yars.nx.Node;
import org.semanticweb.yars.nx.parser.NxParser;

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;

public class ResourceMap {

    String resources;

    public ResourceMap(String resources){
        this.resources = resources;
    }

    public void ntToTSV(){
        try {
            String fileName = this.resources.replace(".nt","_geo_only.tsv");
            File ntFile = new File(fileName);
            ntFile.createNewFile();
            FileWriter writer = new FileWriter(fileName);
            writer.write("WKT\tEntity\n");


            FileInputStream is = new FileInputStream(this.resources);
            NxParser nxp = new NxParser();
            nxp.parse(is);

            HashMap<String,String> mapGeo = new HashMap<>();
            HashMap<String,String> mapWKT = new HashMap<>();

            for (Node[] nx : nxp){
                if(nx[1].toString().equals("<http://www.opengis.net/ont/geosparql#hasGeometry>")){
                    mapGeo.put(nx[2].toString(),nx[0].toString());
                }
                else if(nx[1].toString().equals("<http://www.opengis.net/ont/geosparql#asWKT>")){
                    mapWKT.put(nx[0].toString(),nx[2].toString());
                }
            }

            // Iterating HashMap through for loop
            for (Map.Entry<String, String> set : mapGeo.entrySet()) {
                writer.write(mapWKT.get(set.getKey()) + "\t" + set.getValue() + "\n");
            }


            writer.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void mapEntities(String tsv1, String tsv2){
        try {
            String fileName = this.resources.replace(".nt","_map.nt");
            File ntFile = new File(fileName);
            ntFile.createNewFile();
            FileWriter writer = new FileWriter(fileName);


            FileInputStream is = new FileInputStream(this.resources);
            NxParser nxp = new NxParser();
            nxp.parse(is);

            HashMap<String,String> map1 = this.parseTSV(tsv1,0);
            HashMap<String,String> map2 = this.parseTSV(tsv2,1);

            for (Node[] nx : nxp) {
                String s = map1.get(nx[0].toString());
                String p = nx[1].toString();
                String o = map2.get(nx[2].toString());
                writer.write(s + " " + p + " " + o + " .\n");
            }


            writer.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public HashMap<String,String> parseTSV(String tsv,int index){
        HashMap<String,String> map = new HashMap<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(tsv));
            String line = reader.readLine();
            line = reader.readLine();
            while (line != null) {
                String[] elems = line.split("\t",2);
                map.put("<"+index+">",elems[1]);
                index++;
                line = reader.readLine();
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return map;
    }


    public void keepTripleFromDictionary(String dictionaryFile){
        HashSet<String> set = getDictionary(dictionaryFile);
        if(set==null) return;

        try {
            String fileName = this.resources.replace(".nt","_dict_only.nt");
            File ntFile = new File(fileName);
            ntFile.createNewFile();
            FileWriter writer = new FileWriter(fileName);


            FileInputStream is = new FileInputStream(this.resources);
            NxParser nxp = new NxParser();
            nxp.parse(is);

            for (Node[] nx : nxp){
                if(set.contains(nx[0].toString()) || set.contains(nx[2].toString()))
                    writer.write(nx[0] + " " + nx[1] + " " + nx[2] + " .\n");
            }


            writer.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private HashSet<String> getDictionary(String dictionaryFile){
        HashSet<String> set = new HashSet<>();
        try {
            FileInputStream is = new FileInputStream(dictionaryFile);
            NxParser nxp = new NxParser();
            nxp.parse(is);

            for (Node[] nx : nxp){
                set.add(nx[0].toString());
            }

            return set;
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
            return null;
        }
    }
}
