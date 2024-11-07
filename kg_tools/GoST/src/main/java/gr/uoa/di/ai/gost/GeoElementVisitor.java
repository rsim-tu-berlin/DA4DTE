package gr.uoa.di.ai.gost;

import org.apache.jena.arq.querybuilder.ExprFactory;
import org.apache.jena.graph.Triple;
import org.apache.jena.sparql.core.TriplePath;
import org.apache.jena.sparql.expr.E_Exists;
import org.apache.jena.sparql.expr.E_NotExists;
import org.apache.jena.sparql.expr.Expr;
import org.apache.jena.sparql.expr.ExprFunctionOp;
import org.apache.jena.sparql.syntax.*;

import java.util.List;

public class GeoElementVisitor implements ElementVisitor{

    GeoExprVisitor geoExprVisitor;
    ElementGroup block;
    ElementFilter filter;

    public GeoElementVisitor(){
        geoExprVisitor = new GeoExprVisitor();
        block = new ElementGroup();
        filter = null;
    }

    @Override
    public void visit(ElementTriplesBlock elementTriplesBlock) {
    }

    @Override
    public void visit(ElementPathBlock elementPathBlock) {
        for(TriplePath tp: elementPathBlock.getPattern().getList()){
            if(tp.isTriple()) {
                Triple t = tp.asTriple();
                String predicate = t.getPredicate().toString();
                if (GeoDictionary.getGeometryName(predicate) || GeoDictionary.getWKTName(predicate)) {
                    GeoDictionary.setMapping(t.getObject().toString(), t.getSubject().toString());
                }
            }
        }
        block.addElement(elementPathBlock);
    }

    @Override
    public void visit(ElementFilter elementFilter) {
        Expr expr = elementFilter.getExpr();
        geoExprVisitor.initializeBlock(this.block);
        expr.visit(geoExprVisitor);
    }

    @Override
    public void visit(ElementAssign elementAssign) {

    }

    @Override
    public void visit(ElementBind elementBind) {
    }

    @Override
    public void visit(ElementFind elementFind) {

    }

    @Override
    public void visit(ElementData elementData) {
    }

    @Override
    public void visit(ElementUnion elementUnion) {
        ElementUnion union = new ElementUnion();
        for(Element e:elementUnion.getElements()){
            GeoElementVisitor geoElementVisitor = new GeoElementVisitor();
            e.visit(geoElementVisitor);

            Element element = geoElementVisitor.generateBlock();
            union.addElement(element);
        }

        this.block.addElement(union);
    }

    @Override
    public void visit(ElementOptional elementOptional) {
        GeoElementVisitor geoElementVisitor = new GeoElementVisitor();
        elementOptional.getOptionalElement().visit(geoElementVisitor);

        Element element = geoElementVisitor.generateBlock();
        ElementOptional optional = new ElementOptional(element);
        this.block.addElement(optional);
    }

    @Override
    public void visit(ElementGroup elementGroup) {
        List<Element> elements = elementGroup.getElements();
        for(Element e:elements){
            if(e instanceof ElementFilter) {
                ElementFilter filter = (ElementFilter) e;
                if(filter.getExpr() instanceof E_NotExists || filter.getExpr() instanceof E_Exists) {
                    this.addFunctionOverPattern((ExprFunctionOp)filter.getExpr());
                }
                else {
                    if (this.filter == null)
                        this.filter = (ElementFilter) e;
                    else {
                        ExprFactory factory = new ExprFactory();
                        //Concat all filters into one large filter
                        this.filter = new ElementFilter(factory.and(this.filter.getExpr(), ((ElementFilter) e).getExpr()));
                    }
                }
            }
            else{
                checkMapping(e);
            }
        }
        if(this.filter!=null)
            this.filter.visit(this);
    }

    @Override
    public void visit(ElementDataset elementDataset) {

    }

    @Override
    public void visit(ElementNamedGraph elementNamedGraph) {

    }

    @Override
    public void visit(ElementExists elementExists) {
        GeoElementVisitor geoElementVisitor = new GeoElementVisitor();
        elementExists.getElement().visit(geoElementVisitor);

        Element element = geoElementVisitor.generateBlock();
        ElementExists exists = new ElementExists(element);
        this.block.addElement(exists);
    }

    @Override
    public void visit(ElementNotExists elementNotExists) {
        GeoElementVisitor geoElementVisitor = new GeoElementVisitor();
        elementNotExists.getElement().visit(geoElementVisitor);

        Element element = geoElementVisitor.generateBlock();
        ElementNotExists notExists = new ElementNotExists(element);
        this.block.addElement(notExists);
    }

    @Override
    public void visit(ElementMinus elementMinus) {
        GeoElementVisitor geoElementVisitor = new GeoElementVisitor();
        elementMinus.getMinusElement().visit(geoElementVisitor);

        Element element = geoElementVisitor.generateBlock();
        ElementMinus minus = new ElementMinus(element);
        this.block.addElement(minus);
    }

    @Override
    public void visit(ElementService elementService) {

    }

    @Override
    public void visit(ElementSubQuery elementSubQuery) {
        GoST gost = new GoST(elementSubQuery.getQuery().toString());
        block.addElement(new ElementSubQuery(gost.processQuery()));
    }

    public void addFunctionOverPattern(ExprFunctionOp exprFunctionOp){
        GeoElementVisitor geoElementVisitor = new GeoElementVisitor();
        exprFunctionOp.getElement().visit(geoElementVisitor);

        Element element = geoElementVisitor.generateBlock();
        Expr expr = null;
        if(exprFunctionOp instanceof E_NotExists)
            expr = new E_NotExists(element);
        else
            expr = new E_Exists(element);

        ElementFilter elementFilter = new ElementFilter(expr);
        this.block.addElement(elementFilter);
    }

    private void checkMapping(Element e){
        if(e instanceof ElementPathBlock)
            e.visit(this);
        else if(e instanceof ElementSubQuery)
            e.visit(this);
        else if(e instanceof ElementNotExists || e instanceof ElementExists
                || e instanceof ElementOptional || e instanceof ElementMinus || e instanceof ElementUnion)
            e.visit(this);
        else
            block.addElement(e);
    }

    public Element generateBlock(){
        if(this.filter==null) return this.block;
        else return this.geoExprVisitor.mergeBranches();
    }
}
