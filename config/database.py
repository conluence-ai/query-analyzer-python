# Import necessary libraries
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from dotenv import load_dotenv
from typing import List, Optional, Any, Dict
import threading
from datetime import datetime, timedelta
import logging

# Import constants and mappings
from config.constants import BRAND_TABLE, PRODUCT_TABLE, COLUMN_NAME

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseManager:
    """ A class to manage PostgreSQL database connections and operations with optimization. """
    
    # Class-level cache
    _cache = {}
    _cache_timestamps = {}
    _cache_ttl = timedelta(hours=1)  # Cache validity: 1 hour
    _cache_lock = threading.Lock()
    
    # Connection pool (shared across instances)
    _connection_pool = None
    _pool_lock = threading.Lock()
    
    def __init__(self, use_cache: bool = True, use_connection_pool: bool = True):
        """
            Initialize the DatabaseManager with configuration from environment variables.
            
            Args:
                use_cache (bool): Enable in-memory caching of frequently accessed data
                use_connection_pool (bool): Use connection pooling for better performance
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Set database configuration
        self.host = os.getenv("DB_HOST")
        self.port = int(os.getenv("DB_PORT", 5432))
        self.user = os.getenv("DB_USERNAME")
        self.password = os.getenv("DB_PASSWORD")
        self.dbname = os.getenv("DB_DATABASE")
        
        self.use_cache = use_cache
        self.use_connection_pool = use_connection_pool
        
        # Initialize connection pool if enabled
        if self.use_connection_pool:
            self._initializeConnectionPool()
    
    def _initializeConnectionPool(self):
        """
            Initialize a connection pool for reusing database connections.
            Connection pooling significantly reduces overhead for repeated queries.
        """
        with self._pool_lock:
            if self._connection_pool is None:
                try:
                    self._connection_pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=2,      # Minimum connections in pool
                        maxconn=10,     # Maximum connections in pool
                        host=self.host,
                        port=self.port,
                        user=self.user,
                        password=self.password,
                        dbname=self.dbname
                    )
                    logger.info("Database connection pool initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize connection pool: {e}")
                    self._connection_pool = None
    
    def _getConnection(self) -> psycopg2.extensions.connection:
        """
            Create and return a database connection (from pool if available).
            
            Returns:
                psycopg2.extensions.connection: Active database connection
        """
        if self.use_connection_pool and self._connection_pool:
            try:
                return self._connection_pool.getconn()
            except Exception as e:
                logger.warning(f"Failed to get pooled connection, falling back: {e}")
        
        # Fallback to regular connection
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            dbname=self.dbname,
        )
    
    def _releaseConnection(self, conn: psycopg2.extensions.connection):
        """
            Release connection back to pool or close it.
            
            Args:
                conn: Database connection to release
        """
        if self.use_connection_pool and self._connection_pool:
            try:
                self._connection_pool.putconn(conn)
                return
            except Exception as e:
                logger.warning(f"Failed to return connection to pool: {e}")
        
        # Fallback to closing connection
        try:
            conn.close()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    def _getCacheKey(self, column: str, table: str, where_clause: Optional[str] = None) -> str:
        """
            Generate a unique cache key for a query.
            
            Args:
                column (str): Column name
                table (str): Table name
                where_clause (str, optional): WHERE clause
            
            Returns:
                str: Unique cache key
        """
        return f"{table}:{column}:{where_clause or 'all'}"
    
    def _isCacheValid(self, cache_key: str) -> bool:
        """
            Check if cached data is still valid (not expired).
            
            Args:
                cache_key (str): Cache key to check
            
            Returns:
                bool: True if cache is valid, False otherwise
        """
        if cache_key not in self._cache_timestamps:
            return False
        
        age = datetime.now() - self._cache_timestamps[cache_key]
        return age < self._cache_ttl
    
    def _getFromCache(self, cache_key: str) -> Optional[List[Any]]:
        """
            Retrieve data from cache if available and valid.
            
            Args:
                cache_key (str): Cache key
            
            Returns:
                Optional[List[Any]]: Cached data or None
        """
        with self._cache_lock:
            if cache_key in self._cache and self._isCacheValid(cache_key):
                logger.debug(f"Cache hit: {cache_key}")
                return self._cache[cache_key]
        return None
    
    def _saveToCache(self, cache_key: str, data: List[Any]):
        """
            Save data to cache with timestamp.
            
            Args:
                cache_key (str): Cache key
                data (List[Any]): Data to cache
        """
        with self._cache_lock:
            self._cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now()
            logger.debug(f"Cached: {cache_key} ({len(data)} items)")
    
    def clearCache(self, table: Optional[str] = None):
        """
            Clear cached data (useful after database updates).
            
            Args:
                table (str, optional): Clear cache for specific table only. 
                                     If None, clears all cache.
        """
        with self._cache_lock:
            if table:
                # Clear only cache entries for specific table
                keys_to_delete = [k for k in self._cache.keys() if k.startswith(f"{table}:")]
                for key in keys_to_delete:
                    del self._cache[key]
                    del self._cache_timestamps[key]
                logger.info(f"Cleared cache for table: {table}")
            else:
                # Clear all cache
                self._cache.clear()
                self._cache_timestamps.clear()
                logger.info("Cleared all cache")
    
    def _fetchData(self, column: str, table: str, where_clause: Optional[str] = None, 
                   use_cache: Optional[bool] = None) -> List[Any]:
        """
            Fetch data from a specified column and table in the database.
            Implements caching and optimized queries.
            
            Args:
                column (str): The column name to fetch data from
                table (str): The table name to query
                where_clause (str, optional): SQL WHERE clause without the 'WHERE' keyword.
                use_cache (bool, optional): Override instance cache setting for this query
            
            Returns:
                List[Any]: List of values from the specified column. Returns empty list on error.
        """
        # Determine if we should use cache for this query
        should_cache = use_cache if use_cache is not None else self.use_cache
        
        # Check cache first
        if should_cache:
            cache_key = self._getCacheKey(column, table, where_clause)
            cached_data = self._getFromCache(cache_key)
            if cached_data is not None:
                return cached_data
        
        conn = None
        try:
            # Establish database connection
            conn = self._getConnection()
            cur = conn.cursor()
            
            # Build optimized SQL query
            # Use DISTINCT to avoid duplicates and reduce data transfer
            query = f'SELECT DISTINCT {column} FROM {table}'
            if where_clause:
                query += f' WHERE {where_clause}'
            
            # Add ordering for consistent results (helps with caching)
            query += f' ORDER BY {column}'
            query += ';'
            
            # Execute query
            logger.debug(f"Executing query: {query}")
            cur.execute(query)
            
            # Fetch all results
            rows = cur.fetchall()
            
            # Close cursor
            cur.close()
            
            # Extract column values (filter out None values)
            result = [row[0] for row in rows if row[0] is not None]
            
            # Save to cache
            if should_cache:
                self._saveToCache(cache_key, result)
            
            logger.info(f"Fetched {len(result)} items from {table}.{column}")
            return result
            
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return []
        finally:
            if conn:
                self._releaseConnection(conn)
    
    def _fetchDataBatch(self, column: str, table: str, batch_size: int = 1000, 
                       where_clause: Optional[str] = None) -> List[Any]:
        """
            Fetch data in batches using cursor (server-side cursor for large datasets).
            This prevents loading all data into memory at once.
            
            Args:
                column (str): Column name to fetch
                table (str): Table name
                batch_size (int): Number of rows to fetch per batch
                where_clause (str, optional): WHERE clause
            
            Returns:
                List[Any]: List of all fetched values
        """
        conn = None
        try:
            conn = self._getConnection()
            
            # Use named cursor for server-side cursor (efficient for large datasets)
            cur = conn.cursor(name='fetch_cursor')
            cur.itersize = batch_size
            
            # Build query
            query = f'SELECT DISTINCT {column} FROM {table}'
            if where_clause:
                query += f' WHERE {where_clause}'
            query += f' ORDER BY {column};'
            
            # Execute query
            cur.execute(query)
            
            # Fetch all results in batches
            result = []
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                result.extend([row[0] for row in rows if row[0] is not None])
            
            cur.close()
            logger.info(f"Fetched {len(result)} items in batches from {table}.{column}")
            return result
            
        except psycopg2.Error as e:
            logger.error(f"Database error in batch fetch: {e}")
            return []
        except Exception as e:
            logger.error(f"Error in batch fetch: {e}")
            return []
        finally:
            if conn:
                self._releaseConnection(conn)
    
    def fetchBrandNames(self, use_cache: bool = True) -> List[str]:
        """
            Fetch all brand names from the Brand table.
            Uses caching by default for performance.
            
            Args:
                use_cache (bool): Whether to use cache for this query
            
            Returns:
                List[str]: List of brand names
        """
        return self._fetchData(COLUMN_NAME, BRAND_TABLE, use_cache=use_cache)
    
    def fetchProductNames(self, use_cache: bool = True) -> List[str]:
        """
            Fetch all product names from the product table.
            Uses caching by default for performance.
            
            Args:
                use_cache (bool): Whether to use cache for this query
            
            Returns:
                List[str]: List of product names
        """
        return self._fetchData(COLUMN_NAME, PRODUCT_TABLE, use_cache=use_cache)
    
    def fetchBrandNamesActive(self, use_cache: bool = True) -> List[str]:
        """
            Fetch only active brand names (if you have a status column).
            
            Args:
                use_cache (bool): Whether to use cache
            
            Returns:
                List[str]: List of active brand names
        """
        # Modify WHERE clause based on your schema
        return self._fetchData(COLUMN_NAME, BRAND_TABLE, 
                              where_clause="status = 'active'", 
                              use_cache=use_cache)
    
    def prefetchAllData(self):
        """
            Prefetch and cache all commonly accessed data.
            Call this once during application startup for best performance.
        """
        logger.info("Prefetching all data into cache...")
        self.fetchBrandNames(use_cache=True)
        self.fetchProductNames(use_cache=True)
        logger.info("Prefetch complete")
    
    def closeConnectionPool(self):
        """
            Close all connections in the pool.
            Call this when shutting down the application.
        """
        with self._pool_lock:
            if self._connection_pool:
                self._connection_pool.closeall()
                self._connection_pool = None
                logger.info("Connection pool closed")
    
    def getStats(self) -> Dict[str, Any]:
        """
            Get statistics about cache and connection pool.
            
            Returns:
                Dict with cache size, hit rate, and pool info
        """
        with self._cache_lock:
            return {
                'cache_entries': len(self._cache),
                'cache_tables': list(set(k.split(':')[0] for k in self._cache.keys())),
                'connection_pool_active': self._connection_pool is not None,
            }