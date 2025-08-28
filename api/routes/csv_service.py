# services/csv_service.py
"""
CSV Service Implementation
=========================
Handle CSV parsing, validation, and processing for portfolio data
"""

import csv
import io
import re
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import logging

logger = logging.getLogger(__name__)

# Custom exceptions
class CSVValidationError(Exception):
    """Raised when CSV validation fails"""
    pass

class CSVProcessingError(Exception):
    """Raised when CSV processing fails"""
    pass

@dataclass
class ValidationResult:
    """Results from CSV validation"""
    is_valid: bool
    valid_rows: int
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]

class CSVService:
    """Service for handling CSV operations"""
    
    def __init__(self):
        self.job_statuses = {}  # In production, use Redis or database
        
        # Define expected column mappings
        self.holdings_column_mappings = {
            'ticker': ['ticker', 'symbol', 'stock_symbol', 'security_symbol'],
            'shares': ['shares', 'quantity', 'qty', 'units', 'amount'],
            'purchase_price': ['purchase_price', 'cost_basis', 'price', 'cost', 'avg_cost'],
            'purchase_date': ['purchase_date', 'date', 'buy_date', 'acquired_date'],
            'name': ['name', 'company_name', 'security_name', 'description']
        }
        
        self.transactions_column_mappings = {
            'date': ['date', 'transaction_date', 'trade_date'],
            'type': ['type', 'transaction_type', 'action'],
            'ticker': ['ticker', 'symbol', 'stock_symbol'],
            'shares': ['shares', 'quantity', 'qty'],
            'price': ['price', 'unit_price', 'execution_price'],
            'fee': ['fee', 'commission', 'fees'],
            'description': ['description', 'memo', 'notes']
        }

    def parse_holdings_csv(self, csv_content: str) -> List[Dict[str, Any]]:
        """Parse holdings CSV content into structured data"""
        
        try:
            # Create CSV reader
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            
            # Get headers and normalize
            raw_headers = csv_reader.fieldnames
            if not raw_headers:
                raise CSVValidationError("CSV file is empty or has no headers")
            
            # Map headers to standard names
            header_mapping = self._map_headers(raw_headers, self.holdings_column_mappings)
            
            parsed_data = []
            for row_num, raw_row in enumerate(csv_reader, start=2):  # Start at 2 (header is row 1)
                try:
                    # Map row data using header mapping
                    mapped_row = {}
                    for standard_name, csv_header in header_mapping.items():
                        if csv_header:
                            mapped_row[standard_name] = raw_row.get(csv_header, '').strip()
                    
                    # Skip completely empty rows
                    if not any(mapped_row.values()):
                        continue
                    
                    # Add row metadata
                    mapped_row['_row_number'] = row_num
                    mapped_row['_raw_data'] = raw_row
                    
                    parsed_data.append(mapped_row)
                    
                except Exception as e:
                    logger.warning(f"Error parsing row {row_num}: {e}")
                    # Continue processing other rows
                    continue
            
            return parsed_data
            
        except Exception as e:
            raise CSVProcessingError(f"Failed to parse CSV: {str(e)}")

    def validate_holdings_data(self, parsed_data: List[Dict]) -> ValidationResult:
        """Validate parsed holdings data"""
        
        errors = []
        warnings = []
        recommendations = []
        valid_rows = 0
        
        if not parsed_data:
            errors.append("No data rows found in CSV")
            return ValidationResult(False, 0, errors, warnings, recommendations)
        
        for i, row in enumerate(parsed_data):
            row_num = row.get('_row_number', i + 1)
            row_errors = []
            
            # Required field validation
            ticker = row.get('ticker', '').upper().strip()
            shares = row.get('shares', '').strip()
            
            # Validate ticker
            if not ticker:
                row_errors.append(f"Row {row_num}: Missing required ticker/symbol")
            elif not re.match(r'^[A-Z]{1,5}$', ticker):
                warnings.append(f"Row {row_num}: Ticker '{ticker}' may not be valid format")
            
            # Validate shares
            if not shares:
                row_errors.append(f"Row {row_num}: Missing required shares/quantity")
            else:
                try:
                    shares_num = float(shares)
                    if shares_num <= 0:
                        row_errors.append(f"Row {row_num}: Shares must be positive number")
                    elif shares_num != int(shares_num) and shares_num < 1:
                        warnings.append(f"Row {row_num}: Fractional shares detected ({shares_num})")
                except (ValueError, TypeError):
                    row_errors.append(f"Row {row_num}: Invalid shares value '{shares}'")
            
            # Validate price (optional)
            purchase_price = row.get('purchase_price', '').strip()
            if purchase_price:
                try:
                    price_num = float(purchase_price)
                    if price_num < 0:
                        row_errors.append(f"Row {row_num}: Purchase price cannot be negative")
                    elif price_num > 10000:
                        warnings.append(f"Row {row_num}: High purchase price ${price_num:,.2f}")
                except (ValueError, TypeError):
                    warnings.append(f"Row {row_num}: Invalid price format '{purchase_price}'")
            
            # Validate date (optional)
            purchase_date = row.get('purchase_date', '').strip()
            if purchase_date:
                if not self._parse_date(purchase_date):
                    warnings.append(f"Row {row_num}: Invalid date format '{purchase_date}'")
            
            # If no errors for this row, it's valid
            if not row_errors:
                valid_rows += 1
            else:
                errors.extend(row_errors)
        
        # Add recommendations
        if len(parsed_data) > 100:
            recommendations.append("Large portfolio detected. Consider importing in smaller batches.")
        
        duplicate_tickers = self._find_duplicate_tickers(parsed_data)
        if duplicate_tickers:
            recommendations.append(f"Duplicate tickers found: {', '.join(duplicate_tickers)}. Consider consolidating.")
        
        missing_prices = sum(1 for row in parsed_data if not row.get('purchase_price'))
        if missing_prices > len(parsed_data) * 0.5:
            recommendations.append("Many holdings missing purchase prices. Consider adding for better cost basis tracking.")
        
        is_valid = len(errors) == 0 and valid_rows > 0
        
        return ValidationResult(
            is_valid=is_valid,
            valid_rows=valid_rows,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )

    def parse_transactions_csv(self, csv_content: str) -> List[Dict[str, Any]]:
        """Parse transaction CSV content"""
        
        try:
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            raw_headers = csv_reader.fieldnames
            
            if not raw_headers:
                raise CSVValidationError("CSV file is empty or has no headers")
            
            header_mapping = self._map_headers(raw_headers, self.transactions_column_mappings)
            
            parsed_data = []
            for row_num, raw_row in enumerate(csv_reader, start=2):
                mapped_row = {}
                for standard_name, csv_header in header_mapping.items():
                    if csv_header:
                        mapped_row[standard_name] = raw_row.get(csv_header, '').strip()
                
                if any(mapped_row.values()):
                    mapped_row['_row_number'] = row_num
                    mapped_row['_raw_data'] = raw_row
                    parsed_data.append(mapped_row)
            
            return parsed_data
            
        except Exception as e:
            raise CSVProcessingError(f"Failed to parse transactions CSV: {str(e)}")

    def validate_transactions_data(self, parsed_data: List[Dict]) -> ValidationResult:
        """Validate parsed transaction data"""
        
        errors = []
        warnings = []
        recommendations = []
        valid_rows = 0
        
        for i, row in enumerate(parsed_data):
            row_num = row.get('_row_number', i + 1)
            row_errors = []
            
            # Validate required fields
            date_val = row.get('date', '').strip()
            type_val = row.get('type', '').strip().upper()
            ticker_val = row.get('ticker', '').strip().upper()
            
            if not date_val:
                row_errors.append(f"Row {row_num}: Missing transaction date")
            elif not self._parse_date(date_val):
                row_errors.append(f"Row {row_num}: Invalid date format '{date_val}'")
            
            if not type_val:
                row_errors.append(f"Row {row_num}: Missing transaction type")
            elif type_val not in ['BUY', 'SELL', 'DIVIDEND', 'SPLIT', 'TRANSFER']:
                warnings.append(f"Row {row_num}: Unknown transaction type '{type_val}'")
            
            if not ticker_val and type_val not in ['DIVIDEND', 'TRANSFER']:
                row_errors.append(f"Row {row_num}: Missing ticker for {type_val} transaction")
            
            if not row_errors:
                valid_rows += 1
            else:
                errors.extend(row_errors)
        
        is_valid = len(errors) == 0 and valid_rows > 0
        
        return ValidationResult(
            is_valid=is_valid,
            valid_rows=valid_rows,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )

    def parse_watchlist_csv(self, csv_content: str) -> List[Dict[str, Any]]:
        """Parse watchlist CSV content"""
        
        try:
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            
            parsed_data = []
            for row_num, row in enumerate(csv_reader, start=2):
                if any(row.values()):
                    row['_row_number'] = row_num
                    parsed_data.append(row)
            
            return parsed_data
            
        except Exception as e:
            raise CSVProcessingError(f"Failed to parse watchlist CSV: {str(e)}")

    def validate_watchlist_data(self, parsed_data: List[Dict]) -> ValidationResult:
        """Validate parsed watchlist data"""
        
        errors = []
        warnings = []
        valid_rows = 0
        
        for i, row in enumerate(parsed_data):
            row_num = row.get('_row_number', i + 1)
            
            ticker = row.get('ticker', '').strip().upper()
            if not ticker:
                errors.append(f"Row {row_num}: Missing ticker")
            else:
                valid_rows += 1
        
        is_valid = len(errors) == 0 and valid_rows > 0
        
        return ValidationResult(
            is_valid=is_valid,
            valid_rows=valid_rows,
            errors=errors,
            warnings=warnings,
            recommendations=[]
        )

    def export_holdings_to_csv(self, portfolio, include_market_data=True, include_performance=False) -> str:
        """Export portfolio holdings to CSV format"""
        
        output = io.StringIO()
        
        # Define headers
        headers = ['ticker', 'name', 'shares', 'purchase_price']
        
        if include_market_data:
            headers.extend(['current_price', 'market_value', 'day_change', 'day_change_pct'])
        
        if include_performance:
            headers.extend(['unrealized_gain_loss', 'unrealized_gain_loss_pct', 'allocation_pct'])
        
        writer = csv.writer(output)
        writer.writerow(headers)
        
        # Write holdings data
        for holding in portfolio.holdings:
            row = [
                holding.asset.ticker if holding.asset else '',
                holding.asset.name if holding.asset else '',
                holding.shares,
                holding.purchase_price or ''
            ]
            
            if include_market_data:
                # You'd get current market data here
                current_price = holding.purchase_price or 0  # Placeholder
                market_value = holding.shares * current_price
                row.extend([current_price, market_value, 0, 0])  # Placeholder values
            
            if include_performance:
                # Calculate performance metrics
                unrealized_gain = 0  # Placeholder calculation
                row.extend([unrealized_gain, 0, 0])  # Placeholder values
            
            writer.writerow(row)
        
        csv_content = output.getvalue()
        output.close()
        
        return csv_content

    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of background job"""
        return self.job_statuses.get(job_id)

    def update_job_status(self, job_id: str, status: str, data: Dict):
        """Update job status"""
        if job_id not in self.job_statuses:
            self.job_statuses[job_id] = {
                "created_at": datetime.now().isoformat()
            }
        
        self.job_statuses[job_id].update({
            "status": status,
            "updated_at": datetime.now().isoformat(),
            **data
        })

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a background job"""
        if job_id in self.job_statuses:
            self.job_statuses[job_id]["status"] = "cancelled"
            self.job_statuses[job_id]["updated_at"] = datetime.now().isoformat()
            return True
        return False

    def get_user_import_history(self, user_id: int, limit: int = 10, offset: int = 0, status_filter: Optional[str] = None) -> List[Dict]:
        """Get import history for user"""
        
        # In production, query from database
        # For now, return mock data
        history = [
            {
                "job_id": "csv_import_20240115_143022",
                "filename": "portfolio_holdings.csv",
                "status": "completed",
                "total_rows": 25,
                "success_count": 23,
                "error_count": 2,
                "created_at": "2024-01-15T14:30:22Z",
                "completed_at": "2024-01-15T14:30:45Z"
            }
        ]
        
        if status_filter:
            history = [h for h in history if h["status"] == status_filter]
        
        return history[offset:offset+limit]

    # Private helper methods
    
    def _map_headers(self, raw_headers: List[str], column_mappings: Dict) -> Dict[str, str]:
        """Map CSV headers to standard column names"""
        
        mapping = {}
        used_headers = set()
        
        for standard_name, possible_names in column_mappings.items():
            mapped_header = None
            
            # Try to find exact match first
            for header in raw_headers:
                if header.lower().strip() in [name.lower() for name in possible_names]:
                    if header not in used_headers:
                        mapped_header = header
                        used_headers.add(header)
                        break
            
            # If no exact match, try partial match
            if not mapped_header:
                for header in raw_headers:
                    if header not in used_headers:
                        for possible_name in possible_names:
                            if possible_name.lower() in header.lower():
                                mapped_header = header
                                used_headers.add(header)
                                break
                        if mapped_header:
                            break
            
            mapping[standard_name] = mapped_header
        
        return mapping

    def _parse_date(self, date_string: str) -> Optional[date]:
        """Parse date string in various formats"""
        
        if not date_string:
            return None
        
        # Common date formats to try
        formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%m-%d-%Y',
            '%d/%m/%Y',
            '%d-%m-%Y',
            '%Y/%m/%d',
            '%m/%d/%y',
            '%m-%d-%y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_string.strip(), fmt).date()
            except ValueError:
                continue
        
        return None

    def _find_duplicate_tickers(self, parsed_data: List[Dict]) -> List[str]:
        """Find duplicate tickers in data"""
        
        ticker_counts = {}
        for row in parsed_data:
            ticker = row.get('ticker', '').upper().strip()
            if ticker:
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        
        return [ticker for ticker, count in ticker_counts.items() if count > 1]a