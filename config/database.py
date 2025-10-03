# Import necessary libraries
import os
import psycopg2
from dotenv import load_dotenv
from typing import List, Optional, Any

# Import constants and mappings
from config.constants import BRAND_TABLE, PRODUCT_TABLE, COLUMN_NAME

class DatabaseManager:
    """ A class to manage PostgreSQL database connections and operations. """
    
    def __init__(self):
        """
        Initialize the DatabaseManager with configuration from environment variables.
        
        Loads database credentials from .env file and sets up connection parameters.
        Environment variables required:
            - DB_HOST: Database host address
            - DB_PORT: Database port (default: 5432)
            - DB_USERNAME: Database username
            - DB_PASSWORD: Database password
            - DB_DATABASE: Database name
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Set database configuration
        self.host = os.getenv("DB_HOST")
        self.port = int(os.getenv("DB_PORT", 5432))
        self.user = os.getenv("DB_USERNAME")
        self.password = os.getenv("DB_PASSWORD")
        self.dbname = os.getenv("DB_DATABASE")
        
    def _getConnection(self) -> psycopg2.extensions.connection:
        """
            Create and return a database connection.
            
            Returns:
                psycopg2.extensions.connection: Active database connection
        """
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.dbname,
        )
    
    def fetchData(self, column: str, table: str, where_clause: Optional[str] = None) -> List[Any]:
        """
            Fetch data from a specified column and table in the database.
            
            Args:
                column (str): The column name to fetch data from
                table (str): The table name to query
                where_clause (str, optional): SQL WHERE clause without the 'WHERE' keyword.
                                            Example: "id > 5 AND status = 'active'"
            
            Returns:
                List[Any]: List of values from the specified column. Returns empty list on error.
        """
        try:
            # Establish database connection
            conn = self._getConnection()
            cur = conn.cursor()
            
            # Build SQL query
            query = f'SELECT {column} FROM {table}'
            if where_clause:
                query += f' WHERE {where_clause}'
            query += ';'
            
            # Execute query and fetch results
            cur.execute(query)
            rows = cur.fetchall()
            
            # Close cursor and connection
            cur.close()
            conn.close()
            
            # Return list of column values
            return [row[0] for row in rows]
            
        except psycopg2.Error as e:
            print(f"Database error: {e}")
            return []
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def fetchBrandNames(self) -> List[str]:
        """
            Fetch all brand names from the Brand table.
            
            Returns:
                List[str]: List of brand names
        """
        return self.fetchData(COLUMN_NAME, BRAND_TABLE)
    
    def fetchProductNames(self) -> List[str]:
        """
            Fetch all product names from the product table.
            
            Returns:
                List[str]: List of product names
        """
        return self.fetchData(COLUMN_NAME, PRODUCT_TABLE)
