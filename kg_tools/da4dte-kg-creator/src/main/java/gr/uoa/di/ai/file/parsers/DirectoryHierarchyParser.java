package gr.uoa.di.ai.file.parsers;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class DirectoryHierarchyParser {
    public static void traverse(String dir, String output, FileParser parser){
        if(parser instanceof PoiParser || parser instanceof PortParser || parser instanceof RiverParser) traverseGeneric(dir,output,parser);
        else if(parser instanceof SentinelOneParser || parser instanceof  SentinelTwoParser) traverseSentinel(dir,output,parser);
    }

    private static void traverseGeneric(String path, String output, FileParser parser){
        try {
            File dir = new File(path);
            BufferedWriter writer = new BufferedWriter(new FileWriter(output));
            for (File file : dir.listFiles()) {
                if (file.isDirectory()) {
                    String triples = traverseInner(file,parser);
                    writer.write(triples);
                } else {
                    System.out.println("Ignoring file: " + file.getAbsolutePath());
                }
            }
            writer.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    // Sentinel hierarchy traversal
    private static void traverseSentinel(String path, String output, FileParser parser){
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(output));
            traverseSentinelRecursive(path,output,parser,writer,2);
            writer.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void traverseSentinelRecursive(String path, String output, FileParser parser, BufferedWriter writer,int depth){
        try {
            File dir = new File(path);
            for (File file : dir.listFiles()) {
                if (file.isDirectory()) {
                    if(depth>0) traverseSentinelRecursive(file.getAbsolutePath(),output,parser,writer,depth-1);
                    else if(depth==0){
                        String triples = traverseInner(file, parser);
                        writer.write(triples);
                    }
                } else {
                    System.out.println("Ignoring file: " + file.getAbsolutePath());
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static String traverseInner(File dir, FileParser parser){
        StringBuilder builder = new StringBuilder();
        for(File file: dir.listFiles()){
            builder.append(parser.parse(file.getAbsolutePath()));
        }
        return builder.toString();
    }

}
