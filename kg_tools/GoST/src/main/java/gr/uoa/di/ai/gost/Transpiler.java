package gr.uoa.di.ai.gost;


import org.apache.jena.query.Query;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class Transpiler
{
    public static void main( String[] args ) {
        if(args.length >0 && args.length <= 2){
            GoST gost = new GoST(args[0]);
            Query query = gost.processQuery();
            System.out.println(query.toString());
            if(args.length == 2){
                System.out.println("Writing result to : " + args[1]);
                try {
                    File file = new File(args[1]);
                    file.createNewFile();
                    FileWriter writer = new FileWriter(args[1]);
                    writer.write(query+"\n");
                    writer.close();
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        else{
            System.out.println("Invalid Arguments! Execution format:");
            System.out.println("gr.uoa.di.ai.Transpiler query [output_file]");
        }
    }
}
