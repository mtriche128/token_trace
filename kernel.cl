/**************************************************************************//**
* @file  kernel.cl
* @brief This source file contains the OpenCL kernels.
*****************************************************************************/

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable

/* ------------------------------------------------------------------------- *
 * Define Constants                                                          *
 * ------------------------------------------------------------------------- */

#define LOCAL_SIZE_MAX (128)

/* ------------------------------------------------------------------------- *
 * Define Macros                                                             *
 * ------------------------------------------------------------------------- */


/* ------------------------------------------------------------------------- *
 * Define Types                                                              *
 * ------------------------------------------------------------------------- */

/*
 * Define contour states. These values get assigned to the member 'state' of
 * struct 'token_t'.
 */

#define CS_LEFT  (1 << 0)
#define CS_RIGHT (1 << 1)
#define CS_INNER (1 << 2)
#define CS_OUTER (1 << 3)

/**
 * @brief A token entry.
 */ 

typedef struct __attribute__((__packed__)) TOKEN_ENTRY
{
	uchar  state; // contains flags related to contour type
	uchar  hist;  // pass/hold history for generating chain-codes
	uint   orow;  // contour's origin row coordinate
	uint   ocol;  // contour's origin column coordinate
	uint   id;    // contour identifier
	uint   cx;    // current index in the contour table
} token_t;

/**
 * @brief Execution information about a PE.
 */

typedef struct PE_INFO
{
	uint row;   // row handled by PE(i)
	
	uint ecase; // case handled
	
	bool was_trecv; // token was received
	bool was_tpass; // token was passed
	bool was_theld; // token was held
	bool is_tpx;    // is the current pixel a transition pixel?
	bool is_osp;    // is outer starting-point
	bool is_isp;    // is inner starting point  
	bool is_ep;     // is end-point
	bool curr_px;   // current pixel; true if '1', false otherwise
	
	uchar prev_row_px; // pixel information for the previous row
	uchar row_px;      // pixel information for the current row
	
	token_t touch_token; // a copy of the token touched in a given cycle 
	
	__global token_t *recv_token; // token entry for for receiving
	__global token_t *pass_token; // token entry for passing
	
	token_t *held_token; // entry of the held token
} pe_info_t;

/**
 * @brief Define the contour table.
 */

typedef struct CONTOUR_TABLE
{
	__global uint *data; // contour data
	__global uint *cnt;  // row counter
	uint rows; // number of rows in the table
	uint cols; // number of columns in the table
} ctbl_t;

/* ------------------------------------------------------------------------- *
 * Define Constant Data                                                      *
 * ------------------------------------------------------------------------- */

/* ------------------------------------------------------------------------- *
 * Declare Internal Functions                                                *
 * ------------------------------------------------------------------------- */

void pe_init(pe_info_t *p_info, 
		 uint i,
		 token_t *p_theld, 
		 __global token_t *p_tpass,
		 __global token_t *p_trecv);

void pe_begin(pe_info_t *p_info, uchar r_px, uchar ur_px);
void pe_case1(pe_info_t *p_info, uint row, uint col, ctbl_t *p_tbl);
void pe_case2(pe_info_t *p_info, uint row, uint col, ctbl_t *p_tbl);
void pe_case3(pe_info_t *p_info, uint row, uint col, ctbl_t *p_tbl);
void pe_gencon(pe_info_t *p_info, ctbl_t *p_tbl, uint row, uint col);
void pe_gencon2(pe_info_t *p_info, token_t *p_tkn, ctbl_t *p_tbl, uint row, uint col);
void pe_gencon2_global(pe_info_t *p_info, __global token_t *p_tkn, ctbl_t *p_tbl, uint row, uint col);

void token_move(token_t *src, token_t *dst);
void token_move_global(token_t *src, __global token_t *dst);
void token_global_move(__global token_t *src, token_t *dst);
void token_clear(token_t *trg);
void token_clear_global(__global token_t *trg);
bool token_check(token_t *trg);
bool token_check_global(__global token_t *trg);

void ctbl_append(ctbl_t *p_tbl, token_t *p_tkn, uint row, uint col);
void ctbl_append_global(ctbl_t *p_tbl, __global token_t *p_tkn, uint row, uint col);
void cbtl_term(ctbl_t *p_tbl, token_t *p_tkn);
void cbtl_term_global(ctbl_t *p_tbl, __global token_t *p_tkn);

void print_title(void);
void print_info(pe_info_t *p_info, uint row, uint col, uint t);

/* ------------------------------------------------------------------------- *
 * Define Internal Functions                                                 *
 * ------------------------------------------------------------------------- */

/**
 * @brief Initialize PE state information.
 * 
 * @param p_info  Pointer to PE information.
 * @param i       The index/row of the current PE.
 * @param p_theld Pointer to the held token entry.
 * @param p_tpass Pointer to the pass token entry.
 * @param p_trecv Pointer to the received token entry.
 */

void pe_init(pe_info_t *p_info, 
             uint i,
             token_t *p_theld, 
             __global token_t *p_tpass,
             __global token_t *p_trecv)
{
	p_info->row   = i;
	
	p_info->ecase = 0;
	
	p_info->was_trecv = false;
	p_info->was_tpass = false;
	
	p_info->prev_row_px = 0x00;
	p_info->row_px      = 0x00;
	
	p_info->recv_token = p_trecv;
	p_info->pass_token = p_tpass;
	p_info->held_token = p_theld;
}

/**
 * @brief Execute the first steps in the next PE cycle.
 *
 * @param p_info  The pointer containing PE(i)'s execution info.
 * @param r_px    The right pixel relative to PE(i)'s current position.  
 * @param ur_px   The upper-right pixel relative to PE(i)'s current position.  
 */

void pe_begin(pe_info_t *p_info, uchar r_px, uchar ur_px)
{
	// shift in relative right pixel (look ahead by 1 pixel)
	p_info->row_px = p_info->row_px << 1;
	p_info->row_px |= r_px ? 0x01 : 0x00; // TODO: Find non-branching way to do this.
	
	// shift in relative upper-right pixel
	p_info->prev_row_px = p_info->prev_row_px << 1;
	p_info->prev_row_px |= ur_px ? 0x01 : 0x00; // TODO: Find non-branching way to do this.
	
	p_info->curr_px = p_info->row_px & 0x02;
	
	// check for transition pixel
	if( (p_info->row_px & 0x06) == 0x02 )
	{
		p_info->is_tpx = true;
	}
	else if( (p_info->row_px & 0x03) == 0x02 )
	{
		p_info->is_tpx = true;
	}
	else
	{
		p_info->is_tpx = false;
	}
	
	// check token entries
	if(p_info->recv_token)
		p_info->was_trecv = token_check_global(p_info->recv_token);
	p_info->was_theld = token_check(p_info->held_token);
	
	// which case needs to be handled?
	if( (!(p_info->was_trecv)) && (!(p_info->was_theld)) )
	{
		// neither a token was held or received
		p_info->ecase = 1;
	}
	
	else if(p_info->was_trecv && p_info->was_theld)
	{
		// both a token was held and received
		p_info->ecase = 3;
	}
	
	else if(p_info->was_trecv != p_info->was_theld)
	{
		// either a token was held or received
		p_info->ecase = 2;
	}
	
	else
	{
		p_info->ecase = 0;
		printf("Error: Erronious state detected!\r\n");
	}
	
	// reset these indicators
	p_info->was_tpass = false;
	p_info->is_osp    = false;
	p_info->is_isp    = false;
	p_info->is_ep     = false;
	
	// clear the touch token
	p_info->touch_token.state = 0x00;
}

/**
 * @brief Move a token entry.
 * 
 * @param src source
 * @param dst destination
 */

void token_move(token_t *src, token_t *dst)
{
	dst->state  = src->state;
	dst->orow   = src->orow;
	dst->ocol   = src->ocol;
	dst->hist   = src->hist;
	dst->id     = src->id;
	dst->cx     = src->cx;
	src->state  = 0;
}

void token_move_global(token_t *src, __global token_t *dst)
{
	dst->state  = src->state;
	dst->orow   = src->orow;
	dst->ocol   = src->ocol;
	dst->hist   = src->hist;
	dst->id     = src->id;
	dst->cx     = src->cx;
	src->state  = 0;
}

void token_global_move(__global token_t *src, token_t *dst)
{
	dst->state  = src->state;
	dst->orow   = src->orow;
	dst->ocol   = src->ocol;
	dst->hist   = src->hist;
	dst->id     = src->id;
	dst->cx     = src->cx;
	src->state  = 0;
}

/**
 * @brief Clear token entry. 
 * 
 * After this funtion is executed, the target token entry will be "erased".
 * 
 * @param trg Target token entry.
 */

void token_clear(token_t *trg)
{
	trg->state = 0;
}

void token_clear_global(__global token_t *trg)
{
	trg->state = 0;
}

/**
 * @brief Check if a token entry is loaded (as opposed to cleared).
 * 
 * @param trg Target token entry.
 * 
 * @return True, if the token entry is loaded. False, if the entry is clear.
 */

bool token_check(token_t *trg)
{
	return((bool)trg->state);
}

bool token_check_global(__global token_t *trg)
{
	return(trg->state != 0);
	//return((bool)trg->state);
}

/**
 * @brief Append a contour point.
 * 
 * @param p_tbl Pointer to the contour table.
 * @param p_tkn Pointer to the target token.
 * @param row   The row coordinate of the new contour point.
 * @param col   The col coordinate of the new contour point.
 */

void ctbl_append(ctbl_t *p_tbl, token_t *p_tkn, uint row, uint col)
{
	uint base = p_tkn->id*p_tbl->cols;
	
	// check if there's room to add a new point
	if((p_tkn->cx+1) < p_tbl->cols)
	{
		printf("ctbl_append(): id=%i cx=%i / row=%i col=%i\r\n", p_tkn->id, p_tkn->cx, row, col); 
		p_tbl->data[base+p_tkn->cx++] = row;
		p_tbl->data[base+p_tkn->cx++] = col;
	}
}

void ctbl_append_global(ctbl_t *p_tbl, __global token_t *p_tkn, uint row, uint col)
{
	uint base = p_tkn->id*p_tbl->cols;
	
	// check if there's room to add a new point
	if((p_tkn->cx+1) < p_tbl->cols)
	{
		printf("ctbl_append_global(): id=%i cx=%i / row=%i col=%i\r\n", p_tkn->id, p_tkn->cx, row, col); 
		p_tbl->data[base+p_tkn->cx++] = row;
		p_tbl->data[base+p_tkn->cx++] = col;
	}
}

/**
 * @brief Append a contour point.
 * 
 * @param p_tbl Pointer to the contour table.
 * @param p_tkn Pointer to the target token.
 */

void cbtl_term(ctbl_t *p_tbl, token_t *p_tkn)
{
	uint base = p_tkn->id*p_tbl->cols;
	
	// Record the number of contour coordinates in the first column.
	p_tbl->data[base] = p_tkn->cx; 
}

void cbtl_term_global(ctbl_t *p_tbl, __global token_t *p_tkn)
{
	uint base = p_tkn->id*p_tbl->cols;
	
	// Record the number of contour coordinates in the first column.
	p_tbl->data[base] = p_tkn->cx; 
}

/**
 * @brief Print data table.
 */

void print_title(void)
{
	printf("*--------------------------*----------------------*-----------------------------------*\r\n");
	printf("|      time-space info     |        actions       |             token seen            |\r\n");
	printf("*-------*------*------*----*----------------------*---------------------*------*------*\r\n");
	printf("| PE(i) |    t |  col | px | case osp isp t h r p | origin(r,c) L R O I |   id |   cx |\r\n");
	printf("*-------*------*------*----*----------------------*---------------------*------*------*\r\n");
}

/**
 * @brief Print PE execution info.
 *
 * @param p_info PE execution info.
 * @param row    The current row coordinate.
 * @param col    The current column coordinate.
 * @param t      The current cycle.
 */

#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

void print_info(pe_info_t *p_info, uint row, uint col, uint t)
{	
	if(t%2)
	{
		printf("%s", KCYN);
	}
	
	else
	{
		printf("%s", KNRM);
	}
	
	printf("| %5i | %4i | %4i |  %1i | %3i  %2s  %2s  %s %s %s %s | (%4i,%4i) %s %s %s %s | %4i | %4i |\r\n", 
	       row, t, col, (p_info->row_px & 0x02) >> 1, // time-space
	       // [actions]
	       p_info->ecase,
	       p_info->is_osp ? "x" : "-",
	       p_info->is_isp ? "x" : "-",
	       p_info->is_tpx ? "x" : "-",
	       p_info->was_theld ? "x" : "-",
	       p_info->was_trecv ? "x" : "-",
	       p_info->was_tpass ? "x" : "-",
	       // [token seen]
	       (p_info->ecase == 2) ? p_info->touch_token.orow : -1,
	       (p_info->ecase == 2) ? p_info->touch_token.ocol : -1,
	       (p_info->touch_token.state & CS_LEFT ) ? "x" : "-",
	       (p_info->touch_token.state & CS_RIGHT) ? "x" : "-",
	       (p_info->touch_token.state & CS_OUTER) ? "x" : "-",
	       (p_info->touch_token.state & CS_INNER) ? "x" : "-",
	        p_info->touch_token.state ? p_info->touch_token.id : -1,
	        p_info->touch_token.state ? p_info->touch_token.cx : -1);
}

/**
 * @brief Handle case 1.
 * 
 * @param p_info PE execution info.
 * @param row    The current row coordinate.
 * @param col    The current column coordinate.
 * @param p_tbl  Pointer to the contour table.
 */

void pe_case1(pe_info_t *p_info, uint row, uint col, ctbl_t *p_tbl)
{
	if( ((p_info->row_px & 0x06) == 0x02) && 
	    ((p_info->prev_row_px & 0x07) == 0x00) )
	{
		/* --- An outer starting point was detected. --- */
		
		p_info->is_osp = true;
		
		// check if the passing token entry exists
		if(p_info->pass_token)
		{
			// generate the right token
			p_info->pass_token->state = CS_RIGHT | CS_OUTER;
			p_info->pass_token->hist  = 0;
			p_info->pass_token->orow  = row;
			p_info->pass_token->ocol  = col;
			
			p_info->pass_token->id = atomic_inc(p_tbl->cnt);
			
			/* NOTE: Index 0 (the first column in the contour table) 
			 * shall store the number of appended coordinates within 
			 * the associated row. Thus, it shall be skipped for now
			 * and written when the contour's end-point is found.
			 */
			p_info->pass_token->cx = 1;
			
			p_info->was_tpass = true;
		}
		
		// generate the left token
		p_info->held_token->state = CS_LEFT | CS_OUTER;
		p_info->held_token->hist  = 0;
		p_info->held_token->orow  = row;
		p_info->held_token->ocol  = col;
		
		p_info->held_token->id = atomic_inc(p_tbl->cnt);
		
		/* NOTE: Index 0 (the first column in the contour table) 
		 * shall store the number of appended coordinates within 
		 * the associated row. Thus, it shall be skipped for now
		 * and written when the contour's end-point is found.
		 */
		p_info->held_token->cx = 1;
		
		p_info->was_theld = true;
	}
	
	/*
	 * TODO: Detect inner starting point. */
	else if( ((p_info->row_px & 0x06) == 0x04) && 
		   ((p_info->prev_row_px & 0x07) == 0x07) )
	{
		// --- An inner starting point was detected. --- 
		
		p_info->is_isp = true;
		
		// check if the passing token entry exists
		if(p_info->pass_token)
		{
			// generate the right token
			p_info->pass_token->state = CS_RIGHT | CS_INNER;
			p_info->pass_token->hist  = 0;
			p_info->pass_token->orow  = row;
			p_info->pass_token->ocol  = col;
			
			p_info->pass_token->id = atomic_inc(p_tbl->cnt);
			
			/* NOTE: Index 0 (the first column in the contour table) 
			 * shall store the number of appended coordinates within 
			 * the associated row. Thus, it shall be skipped for now
			 * and written when the contour's end-point is found.
			 */
			p_info->pass_token->cx = 1;
			
			p_info->was_tpass = true;
		}
		
		// generate the left token
		p_info->held_token->state = CS_LEFT | CS_INNER;
		p_info->held_token->hist  = 0;
		p_info->held_token->orow  = row;
		p_info->held_token->ocol  = col;
		
		p_info->held_token->id = atomic_inc(p_tbl->cnt);
		
		/* NOTE: Index 0 (the first column in the contour table) 
		 * shall store the number of appended coordinates within 
		 * the associated row. Thus, it shall be skipped for now
		 * and written when the contour's end-point is found.
		 */
		p_info->held_token->cx = 1;
		
		p_info->was_theld = true;
	}
	
	if(p_info->held_token->state)
	{
		pe_gencon2(p_info, p_info->held_token, p_tbl, row, col);
	}
	
	if(p_info->pass_token->state)
	{
		pe_gencon2_global(p_info, p_info->pass_token, p_tbl, row, col);
	}
}

/**
 * @brief Handle case 2.
 * 
 * @param p_info PE execution info.
 * @param row    The current row coordinate.
 * @param col    The current column coordinate.
 */

void pe_case2(pe_info_t *p_info, uint row, uint col, ctbl_t *p_tbl)
{
	uchar pass = 1;
	
	p_info->held_token->hist = p_info->held_token->hist << 1; 
	
	// If a token was received, hold it.
	if(p_info->was_trecv)
	{
		token_global_move(p_info->recv_token, p_info->held_token);
		p_info->held_token->hist |= 0x01;
	}
	
	// record the token touched by the PE this cycle
	p_info->touch_token.state = p_info->held_token->state;
	p_info->touch_token.hist  = p_info->held_token->hist;
	p_info->touch_token.orow  = p_info->held_token->orow;
	p_info->touch_token.ocol  = p_info->held_token->ocol;
	p_info->touch_token.id    = p_info->held_token->id;
	p_info->touch_token.cx    = p_info->held_token->cx;
	
	pe_gencon2(p_info, p_info->held_token, p_tbl, row, col);
	
	if(p_info->row_px & 0x02)
	{
		// The current pixel is '1'.
		
		if( (p_info->held_token->state == (CS_OUTER | CS_LEFT )) ||
		    (p_info->held_token->state == (CS_INNER | CS_RIGHT)) )
		{
			pass = 0;
		}
	}
	
	else
	{
		// The current pixel is '0'.
		
		if( (p_info->held_token->state == (CS_INNER | CS_LEFT )) ||
		    (p_info->held_token->state == (CS_OUTER | CS_RIGHT)) )
		{
			pass = 0;
		}
	}
	
	if(pass)
	{
		token_move_global(p_info->held_token, p_info->pass_token);
		p_info->was_tpass = true;
	}
}

/**
 * @brief Handle case 3.
 * 
 * @param p_info PE execution info.
 * @param row    The current row coordinate.
 * @param col    The current column coordinate.
 */

void pe_case3(pe_info_t *p_info, uint row, uint col, ctbl_t *p_tbl)
{
	p_info->is_ep = true;
	p_info->recv_token->state = 0;
	p_info->held_token->state = 0;

	pe_gencon2(p_info, p_info->held_token, p_tbl, row, col);
	pe_gencon2_global(p_info, p_info->recv_token, p_tbl, row, col);
}

/**
 * @brief Generate contour information.
 * 
 * @param p_info Pointer to PE execution info.
 * @param p_tbl  Pointer to the contour table.
 * @param row    Current pixel row.
 * @param col    Current pixel col.
 */

void pe_gencon(pe_info_t *p_info, ctbl_t *p_tbl, uint row, uint col)
{
	if(p_info->ecase == 1)
	{
		if(p_info->is_osp)
		{
			ctbl_append(p_tbl, &(p_info->touch_token), row, col);
		}
		
		else if(p_info->is_isp)
		{
			ctbl_append(p_tbl, &(p_info->touch_token), row-1, col);
			
			if(col != 0)
			{
				ctbl_append(p_tbl, &(p_info->touch_token), row, col-1);
			}
		}
	}
	
	else if(p_info->ecase == 2)
	{
		if(p_info->is_tpx)
		{
			ctbl_append(p_tbl, &(p_info->touch_token), row, col);
		}
		
		// check for chain-code 4
		if( (p_info->touch_token.state == (CS_OUTER | CS_LEFT)) ||
			(p_info->touch_token.state == (CS_INNER | CS_RIGHT)) )
		{
			if( ((p_info->row_px & 0x06) == 0x06) &&
				((p_info->touch_token.hist & 0x01) == 0x00) )
			{
				ctbl_append(p_tbl, &(p_info->touch_token), row, col);
			}
		}
		
		// check for chain-code 0
		if(p_info->touch_token.state == (CS_OUTER | CS_RIGHT))
		{
			if( ((p_info->row_px & 0x0E) == 0x00) &&
				((p_info->touch_token.hist & 0x03) == 0x00) )
			{
				ctbl_append(p_tbl, &(p_info->touch_token), row-1, col);
			}
		}
		
		else if(p_info->touch_token.state == (CS_INNER | CS_LEFT))
		{
			if( ((p_info->row_px & 0x06) == 0x00) &&
				((p_info->touch_token.hist & 0x01) == 0x00) )
			{
				ctbl_append(p_tbl, &(p_info->touch_token), row-1, col);
			}
		}
	}
	
	else if(p_info->ecase == 3)
	{
		if(p_info->is_ep)
		{
			//if(p_info->touch_token.state & CS_INNER)
			//{
			ctbl_append(p_tbl, &(p_info->touch_token), row, col);
			cbtl_term(p_tbl, &(p_info->touch_token)); // terminate the current contour chain
			//}
		}
	}
}

/**
 * @brief Generate contour information.
 * 
 * @param p_info Pointer to PE execution info.
 * @param p_info Pointer to the target token.
 * @param p_tbl  Pointer to the contour table.
 * @param row    Current pixel row.
 * @param col    Current pixel col.
 */

void pe_gencon2(pe_info_t *p_info, token_t *p_tkn, ctbl_t *p_tbl, uint row, uint col)
{
	printf("pe_gencon2()\r\n");
	
	if(p_info->ecase == 1)
	{
		if(p_info->is_osp)
		{
			ctbl_append(p_tbl, p_tkn, row, col);
		}
		
		else if(p_info->is_isp)
		{
			ctbl_append(p_tbl, p_tkn, row-1, col);
			
			if(col != 0)
			{
				ctbl_append(p_tbl, p_tkn, row, col-1);
			}
		}
	}
	
	else if(p_info->ecase == 2)
	{
		if(p_info->is_tpx)
		{
			printf("1 ");
			ctbl_append(p_tbl, p_tkn, row, col);
		}
		
		// check for chain-code 4
		else if( (p_tkn->state == (CS_OUTER | CS_LEFT)) ||
			(p_tkn->state == (CS_INNER | CS_RIGHT)) )
		{
			if( ((p_info->row_px & 0x06) == 0x06) &&
				((p_tkn->hist & 0x01) == 0x00) )
			{
				printf("2 ");
				ctbl_append(p_tbl, p_tkn, row, col);
			}
		}
		
		// check for chain-code 0
		else if(p_tkn->state == (CS_OUTER | CS_RIGHT))
		{
			if( ((p_info->row_px & 0x0E) == 0x00) &&
				((p_tkn->hist & 0x03) == 0x00) )
			{
				printf("3 ");
				ctbl_append(p_tbl, p_tkn, row-1, col);
			}
		}
		
		else if(p_tkn->state == (CS_INNER | CS_LEFT))
		{
			if( ((p_info->row_px & 0x06) == 0x00) &&
				((p_tkn->hist & 0x01) == 0x00) )
			{
				printf("4 ");
				ctbl_append(p_tbl, p_tkn, row-1, col);
			}
		}
	}
	
	else if(p_info->ecase == 3)
	{
		if(p_info->is_ep)
		{
			if(p_info->curr_px)
			{
				printf("8 ");
				ctbl_append(p_tbl, p_tkn, row, col);
				cbtl_term(p_tbl, p_tkn); // terminate the current contour chain
			}
			
			else
			{
				printf("9 ");
				ctbl_append(p_tbl, p_tkn, row-1, col);
				cbtl_term(p_tbl, p_tkn); // terminate the current contour chain
			}
		}
	}
}

void pe_gencon2_global(pe_info_t *p_info, __global token_t *p_tkn, ctbl_t *p_tbl, uint row, uint col)
{
	printf("pe_gencon2_global()\r\n");
	
	if(p_info->ecase == 1)
	{
		if(p_info->is_osp)
		{
			printf("1 ");
			ctbl_append_global(p_tbl, p_tkn, row, col);
		}
		
		else if(p_info->is_isp)
		{
			printf("2 ");
			ctbl_append_global(p_tbl, p_tkn, row-1, col);
			
			if(col != 0)
			{
				printf("3 ");
				ctbl_append_global(p_tbl, p_tkn, row, col-1);
			}
		}
	}
	
	else if(p_info->ecase == 2)
	{
		if(p_info->is_tpx)
		{
			printf("4 ");
			ctbl_append_global(p_tbl, p_tkn, row, col);
		}
		
		// check for chain-code 4
		if( (p_tkn->state == (CS_OUTER | CS_LEFT)) ||
			(p_tkn->state == (CS_INNER | CS_RIGHT)) )
		{
			if( ((p_info->row_px & 0x06) == 0x06) &&
				((p_tkn->hist & 0x01) == 0x00) )
			{
				printf("5 ");
				ctbl_append_global(p_tbl, p_tkn, row, col);
			}
		}
		
		// check for chain-code 0
		if(p_tkn->state == (CS_OUTER | CS_RIGHT))
		{
			if( ((p_info->row_px & 0x0E) == 0x00) &&
				((p_tkn->hist & 0x03) == 0x00) )
			{
				printf("6 ");
				ctbl_append_global(p_tbl, p_tkn, row-1, col);
			}
		}
		
		else if(p_tkn->state == (CS_INNER | CS_LEFT))
		{
			if( ((p_info->row_px & 0x06) == 0x00) &&
				((p_tkn->hist & 0x01) == 0x00) )
			{
				printf("7 ");
				ctbl_append_global(p_tbl, p_tkn, row-1, col);
			}
		}
	}
	
	else if(p_info->ecase == 3)
	{
		if(p_info->is_ep)
		{
			if(p_info->curr_px)
			{
				printf("8 ");
				ctbl_append_global(p_tbl, p_tkn, row, col);
				cbtl_term_global(p_tbl, p_tkn); // terminate the current contour chain
			}
			
			else
			{
				
				printf("9 ");
				ctbl_append_global(p_tbl, p_tkn, row-1, col);
				cbtl_term_global(p_tbl, p_tkn); // terminate the current contour chain
			}
		}
	}
}

/* ------------------------------------------------------------------------- *
 * Define Kernels                                                            *
 * ------------------------------------------------------------------------- */

__kernel void TOKEN_TRACE ( __global uchar *bin_img,
				    __global token_t *token_table,
				    const uint rows,
				    const uint cols,
				    __global uint *ctbl_cnt,
				    __global uint *ctbl_data,
				    const uint ctbl_rows,
				    const uint ctbl_cols)
{
	
	unsigned int local_id = get_local_id(0);
	unsigned int row = get_global_id(0);
	unsigned int col = 0;
	
	__global unsigned char *bin_img_prev_row = bin_img + cols*(row-1);
	__global unsigned char *bin_img_row = bin_img + cols*row;
	
	const unsigned int T = 2*rows + cols; // total cycles which will be executed
	unsigned int t; // stores current cycle
	
	// define and initialize the token held by PE(i)
	token_t held_token = {
		.state = 0, // if 0, then no token held
		.orow  = 0,
		.ocol  = 0
	};
	
	// ------------------------------------------------------------
	// Initialize the token table.
	
	token_table[row].state = 0;
	token_table[row].orow  = 0;
	token_table[row].ocol  = 0;
	
	// ------------------------------------------------------------
	// Initialize the contour table.
	
	ctbl_t ctbl = {
		.data = ctbl_data,
		.cnt  = ctbl_cnt,
		.rows = ctbl_rows,
		.cols = ctbl_cols
	};
	
	// ------------------------------------------------------------
	// Initialize PE state.
	
	pe_info_t info;
	
	pe_init(&info, 
	       row, 
	       &held_token, 
	       (row < rows) ? (token_table+row+1) : 0,
	       token_table+row); 
	
	// ------------------------------------------------------------
	
	if(row == 0)
	{ 
		printf("Contour Table: rows=%i cols=%i init(cnt)=%i\r\n", ctbl.rows, ctbl.cols, *ctbl.cnt);
		printf("Total Cycles = %i\r\n", T);	
		print_title();
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
	
	for(t = 0; t < T; t++)
	{
		/* ------ ignore PE(i) if it's not processing data ------ */
		
		if( (row >= rows) || (col >= cols) )
		{
			barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
			barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
			continue;
		}
		
		/* ------ stall PE(i) by [2*row] cycles (skew) ----- */
		
		if(t < 2*row)
		{
			barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
			barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
			continue;
		}
		
		/* ------ execute next cycle for PE(i) ----- */
		
		pe_begin(&info, 
			   (col < (cols-1)) ? bin_img_row[col+1] : 0, 
			   (row != 0) ? bin_img_prev_row[col] : 0);
		
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
		
		switch(info.ecase)
		{
			case 1:
				pe_case1(&info, row, col, &ctbl);
				break;
				
			case 2:
				pe_case2(&info, row, col, &ctbl);
				break;
				
			case 3:
				pe_case3(&info, row, col, &ctbl);
				break;
		}
		
		print_info(&info, row, col, t);
		//pe_gencon(&info,&ctbl,row,col);
		
		barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
		
		col++;
	}
}
