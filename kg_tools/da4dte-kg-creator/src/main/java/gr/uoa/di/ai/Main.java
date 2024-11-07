package gr.uoa.di.ai;

import gr.uoa.di.ai.file.parsers.*;

public class Main {
    public static void main(String[] args) {
        DirectoryHierarchyParser.traverse(args[0],args[1], new PoiParser());
        DirectoryHierarchyParser.traverse(args[2],args[3], new PortParser());
        DirectoryHierarchyParser.traverse(args[4],args[5], new RiverParser());
        DirectoryHierarchyParser.traverse(args[6],args[7], new SentinelTwoParser());
    }
}