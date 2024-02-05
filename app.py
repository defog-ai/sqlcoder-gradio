import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model = AutoModelForCausalLM.from_pretrained("defog/sqlcoder-7b-2", torch_dtype = torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")

pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

def predict(question, ddl):
    prompt = f"""### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]
### Database Schema
The query will run on a database with the following schema:
{ddl}
### Answer
Given the database schema, here is the SQL query that [QUESTION]{question}[/QUESTION]
[SQL]
"""
    predictions = pipeline(
        row["prompt"],
        max_new_tokens=300,
        do_sample=False,
        num_beams=4,
        num_return_sequences=1,
        return_full_text=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"].split("```")[0].split(";")[0].strip()+ ";"
    return predictions

gradio_app = gr.Interface(
    fn=predict,
    inputs=[
        gradio.Textbox(lines=2, value="What are our top 3 products by revenue in the New York region?", label="question"),
        gradio.Textbox(lines=20, value="""CREATE TABLE products (
  product_id INTEGER PRIMARY KEY, -- Unique ID for each product
  name VARCHAR(50), -- Name of the product
  price DECIMAL(10,2), -- Price of each unit of the product
  quantity INTEGER  -- Current quantity in stock
);
CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
  name VARCHAR(50), -- Name of the customer
  address VARCHAR(100) -- Mailing address of the customer
);
CREATE TABLE salespeople (
  salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson 
  name VARCHAR(50), -- Name of the salesperson
  region VARCHAR(50) -- Geographic sales region 
);
CREATE TABLE sales (
  sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
  product_id INTEGER, -- ID of product sold
  customer_id INTEGER,  -- ID of customer who made purchase
  salesperson_id INTEGER, -- ID of salesperson who made the sale
  sale_date DATE, -- Date the sale occurred 
  quantity INTEGER -- Quantity of product sold
);
CREATE TABLE product_suppliers (
  supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
  product_id INTEGER, -- Product ID supplied
  supply_price DECIMAL(10,2) -- Unit price charged by supplier
);
-- sales.product_id can be joined with products.product_id
-- sales.customer_id can be joined with customers.customer_id 
-- sales.salesperson_id can be joined with salespeople.salesperson_id
-- product_suppliers.product_id can be joined with products.product_id""", label="database schema"),
    ],
    outputs="text",
    title="Convert an English question into SQL",
)

if __name__ == "__main__":
    gradio_app.launch()